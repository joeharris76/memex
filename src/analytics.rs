use crate::state::SessionScope;
use crate::types::{
    Record, SourceFilter, SourceKind, jcode_text_is_subagent_directive,
    jcode_tmp_cwd_is_worker_sandbox,
};
use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params, params_from_iter};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const SCHEMA_VERSION: i64 = 9;
const GIT_METADATA_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_LABEL_CHARS: usize = 150;
pub const UNFILED_PROJECT: &str = "Unfiled";
const REPOSITORY_PROJECT_SQL: &str = "COALESCE(NULLIF(repo_project, ''), 'Unfiled')";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectGrouping {
    #[default]
    Flat,
    Repository,
}

#[derive(Clone, Debug)]
pub struct SessionRow {
    pub source: SourceKind,
    pub session_id: String,
    pub source_path: String,
    pub project: String,
    pub display_project: String,
    pub cwd: Option<String>,
    pub last_at: u64,
    pub message_count: u64,
    pub label: Option<String>,
    pub conversation_kind: Option<String>,
}

/// Full-index repository totals, keyed exactly like the sessions project filter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectSummary {
    pub project: String,
    pub session_count: u64,
    pub last_at: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionKindFilter {
    Primary,
    Subagent,
    #[default]
    Regular,
    All,
}

impl SessionKindFilter {
    /// Shared origin predicate. Regular includes ordinary sessions of every
    /// kind; permission reviews require the explicit All filter.
    pub fn matches_kind(self, kind: Option<&str>) -> bool {
        match self {
            SessionKindFilter::All => true,
            SessionKindFilter::Regular => kind != Some("guardian_review"),
            SessionKindFilter::Primary => kind.is_none() || kind == Some("main"),
            SessionKindFilter::Subagent => {
                kind.is_some() && kind != Some("main") && kind != Some("guardian_review")
            }
        }
    }
}

/// A session row with every stored column, for `memex sessions`.
#[derive(Clone, Debug, Serialize)]
pub struct SessionDetailRow {
    pub source: SourceKind,
    pub session_id: String,
    pub source_path: String,
    pub project: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_root: Option<String>,
    pub started_at: u64,
    pub last_at: u64,
    pub message_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_kind: Option<String>,
}

pub struct AnalyticsStore {
    conn: Connection,
}

pub struct AnalyticsWriter {
    store: AnalyticsStore,
    sessions: HashMap<SessionKey, SessionAccumulator>,
    metadata_cache: HashMap<SessionKey, SessionMetadata>,
    git_cache: HashMap<String, GitMetadata>,
    cwd_overrides: HashMap<SessionKey, String>,
    opencode_cache: OpencodeLookupCache,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SessionKey {
    source: SourceKind,
    session_id: String,
    source_path: String,
}

#[derive(Clone, Debug)]
struct SessionAccumulator {
    key: SessionKey,
    project: String,
    started_at: u64,
    last_at: u64,
    message_count: u64,
    first_user_text: Option<String>,
    conversation_kind: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct SessionMetadata {
    pub cwd: Option<String>,
    pub git_root: Option<String>,
    pub git_common_dir: Option<String>,
    pub repo_project: Option<String>,
    pub resolution_status: String,
}

impl AnalyticsStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.busy_timeout(Duration::from_secs(2))?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(Duration::from_secs(2))?;
        conn.pragma_update(None, "query_only", true)?;
        Ok(Self { conn })
    }

    fn init(&self) -> Result<()> {
        // Ingest state advances only after analytics commits. FULL keeps each WAL commit durable
        // before the cross-store publication can clear its recovery marker.
        self.conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = FULL;
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                source TEXT NOT NULL,
                session_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                project TEXT NOT NULL,
                cwd TEXT,
                git_root TEXT,
                git_common_dir TEXT,
                repo_project TEXT,
                started_at INTEGER NOT NULL,
                last_at INTEGER NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                resolution_status TEXT NOT NULL DEFAULT '',
                label TEXT,
                conversation_kind TEXT,
                PRIMARY KEY (source, session_id, source_path)
            );
            CREATE INDEX IF NOT EXISTS sessions_last_at_idx ON sessions(last_at);
            CREATE INDEX IF NOT EXISTS sessions_project_last_at_idx ON sessions(project, last_at);
            CREATE INDEX IF NOT EXISTS sessions_repo_project_last_at_idx ON sessions(repo_project, last_at);
            DROP INDEX IF EXISTS sessions_display_project_last_at_idx;
            CREATE INDEX IF NOT EXISTS sessions_repository_project_last_at_idx
                ON sessions(COALESCE(NULLIF(repo_project, ''), 'Unfiled'), last_at);
            CREATE INDEX IF NOT EXISTS sessions_source_last_at_idx ON sessions(source, last_at);
            "#,
        )?;
        // Additive migrations for existing databases: ignore duplicate-column errors.
        for sql in [
            "ALTER TABLE sessions ADD COLUMN label TEXT",
            "ALTER TABLE sessions ADD COLUMN conversation_kind TEXT",
        ] {
            let _ = self.conn.execute(sql, []);
        }
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS sessions_conversation_kind_idx ON sessions(conversation_kind);
             CREATE INDEX IF NOT EXISTS sessions_label_idx ON sessions(label);
             CREATE INDEX IF NOT EXISTS sessions_git_root_idx ON sessions(git_root);
             CREATE INDEX IF NOT EXISTS sessions_git_common_dir_idx ON sessions(git_common_dir);",
        )?;
        let previous_schema_version: Option<i64> = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .and_then(|value| value.parse().ok());
        if previous_schema_version != Some(SCHEMA_VERSION) {
            self.conn
                .execute("DELETE FROM meta WHERE key = 'analytics_complete'", [])?;
            // Labels generated before the system-tag stripper was complete contained raw
            // system wrappers and truncated prefixes. Clear them so the next backfill
            // recomputes with comprehensive stripping and suffix-preserving truncation.
            let _ = self.conn.execute(
                "UPDATE sessions SET label = NULL WHERE \
                 label LIKE '%<system-reminder>%' OR \
                 label LIKE '%<command-message>%' OR \
                 label LIKE '%<command-name>%' OR \
                 label LIKE '%<INSTRUCTIONS>%' OR \
                 label LIKE '%<environment_context>%' OR \
                 label LIKE '%<recommended_plugins>%' OR \
                 label LIKE '%<user_instructions>%' OR \
                 label LIKE '%<skill>%' OR \
                 label LIKE '%<%' OR \
                 label LIKE '# AGENTS.md%' OR \
                 label LIKE 'You are a reminder observer%'",
                [],
            );
            if previous_schema_version.unwrap_or(0) < 9 {
                let _ = self.migrate_v9_repo_projects();
            }
        }
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![SCHEMA_VERSION.to_string()],
        )?;
        Ok(())
    }

    fn migrate_v9_repo_projects(&self) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "SELECT source, session_id, source_path, project, cwd FROM sessions
             WHERE repo_project IS NULL OR repo_project = '' OR repo_project = '.codex'",
        )?;
        let rows: Vec<(String, String, String, String, Option<String>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })?
            .filter_map(Result::ok)
            .collect();
        drop(stmt);

        let mut git_cache: HashMap<String, GitMetadata> = HashMap::new();
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut update = tx.prepare(
                "UPDATE sessions SET repo_project = ?1, git_root = COALESCE(git_root, ?2), git_common_dir = COALESCE(git_common_dir, ?3), cwd = COALESCE(cwd, ?4)
                 WHERE source = ?5 AND session_id = ?6 AND source_path = ?7",
            )?;
            for (source_str, session_id, source_path, _project, cwd) in rows {
                let source = SourceKind::from_label(&source_str);
                let resolved_cwd = cwd.or_else(|| {
                    if source == Some(SourceKind::Claude) {
                        claude_cwd_from_source_path(&source_path)
                    } else {
                        None
                    }
                });
                if let Some(ref c) = resolved_cwd {
                    let git = git_cache
                        .entry(c.clone())
                        .or_insert_with(|| git_metadata_for_cwd(c));
                    if let Some(ref repo_project) = git.repo_project {
                        let _ = update.execute(params![
                            repo_project,
                            git.git_root,
                            git.git_common_dir,
                            resolved_cwd,
                            source_str,
                            session_id,
                            source_path,
                        ]);
                    }
                }
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn session_count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))?;
        Ok(count.max(0) as u64)
    }

    pub fn is_ready(path: impl AsRef<Path>) -> bool {
        Self::open_read_only(path)
            .and_then(|store| store.session_count())
            .map(|count| count > 0)
            .unwrap_or(false)
    }

    pub fn is_complete(path: impl AsRef<Path>) -> bool {
        Self::open_read_only(path)
            .and_then(|store| store.complete())
            .unwrap_or(false)
    }

    pub fn complete(&self) -> Result<bool> {
        let value: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'analytics_complete'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value.as_deref() == Some("1"))
    }

    pub fn mark_complete(&self) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('analytics_complete', '1')
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [],
        )?;
        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        self.conn.execute("DELETE FROM sessions", [])?;
        Ok(())
    }

    pub fn delete_source_path(&self, source_path: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM sessions WHERE source_path = ?1",
            params![source_path],
        )?;
        Ok(())
    }

    pub fn delete_session_scope(&self, scope: &SessionScope) -> Result<()> {
        self.conn.execute(
            "DELETE FROM sessions WHERE source = ?1 AND source_path = ?2 AND session_id = ?3",
            params![
                SourceKind::Opencode.storage_label(),
                scope.source_path,
                scope.session_id
            ],
        )?;
        Ok(())
    }

    pub fn query_sessions(
        &self,
        source: Option<SourceFilter>,
        since_ms: Option<u64>,
        project: Option<&str>,
        grouping: ProjectGrouping,
        limit: Option<usize>,
    ) -> Result<Vec<SessionRow>> {
        self.query_sessions_filtered(source, since_ms, project, grouping, None, limit)
    }

    pub fn query_sessions_filtered(
        &self,
        source: Option<SourceFilter>,
        since_ms: Option<u64>,
        project: Option<&str>,
        grouping: ProjectGrouping,
        kind: Option<SessionKindFilter>,
        limit: Option<usize>,
    ) -> Result<Vec<SessionRow>> {
        let mut sql = format!(
            "SELECT source, session_id, source_path, project,
                    {REPOSITORY_PROJECT_SQL} AS display_project,
                    cwd, last_at, message_count, label, conversation_kind
             FROM sessions"
        );
        let mut clauses = Vec::new();
        let mut values: Vec<rusqlite::types::Value> = Vec::new();

        if let Some(source) = source {
            let labels = source.storage_labels();
            let placeholders = std::iter::repeat_n("?", labels.len())
                .collect::<Vec<_>>()
                .join(", ");
            clauses.push(format!("source IN ({placeholders})"));
            values.extend(
                labels
                    .iter()
                    .map(|label| rusqlite::types::Value::Text((*label).to_string())),
            );
        }
        if let Some(since_ms) = since_ms {
            clauses.push("last_at >= ?".to_string());
            values.push(rusqlite::types::Value::Integer(since_ms as i64));
        }
        if let Some(project) = project {
            match grouping {
                ProjectGrouping::Flat => clauses.push("project = ?".to_string()),
                ProjectGrouping::Repository => {
                    clauses.push(format!("{REPOSITORY_PROJECT_SQL} = ?"))
                }
            }
            values.push(rusqlite::types::Value::Text(project.to_string()));
        }
        if let Some(kind) = kind {
            match kind {
                SessionKindFilter::Primary => {
                    clauses.push(
                        "(conversation_kind IS NULL OR conversation_kind = 'main')".to_string(),
                    );
                }
                SessionKindFilter::Subagent => {
                    clauses.push(
                        "conversation_kind IS NOT NULL AND conversation_kind NOT IN ('main', 'guardian_review')".to_string(),
                    );
                }
                SessionKindFilter::Regular => {
                    clauses.push(
                        "(conversation_kind IS NULL OR conversation_kind != 'guardian_review')"
                            .to_string(),
                    );
                }
                SessionKindFilter::All => {}
            }
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(" ORDER BY last_at DESC");
        if let Some(limit) = limit {
            sql.push_str(" LIMIT ?");
            values.push(rusqlite::types::Value::Integer(limit as i64));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            let source_label: String = row.get(0)?;
            let source = SourceKind::from_label(&source_label).unwrap_or(SourceKind::Claude);
            let project: String = row.get(3)?;
            let raw_display_project: String = match grouping {
                ProjectGrouping::Flat => project.clone(),
                ProjectGrouping::Repository => row.get(4)?,
            };
            let display_project = display_project_name(&raw_display_project);
            Ok(SessionRow {
                source,
                session_id: row.get(1)?,
                source_path: row.get(2)?,
                project,
                display_project,
                cwd: row.get::<_, Option<String>>(5)?.filter(|v| !v.is_empty()),
                last_at: row.get::<_, i64>(6)?.max(0) as u64,
                message_count: row.get::<_, i64>(7)?.max(0) as u64,
                label: row.get::<_, Option<String>>(8)?.filter(|v| !v.is_empty()),
                conversation_kind: row.get::<_, Option<String>>(9)?.filter(|v| !v.is_empty()),
            })
        })?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Sessions with full stored metadata, newest first. `cwd` restricts to
    /// sessions whose working directory is the given path, lives under it,
    /// or whose git root is the given path (so a repo path matches sessions
    /// started in any of its subdirectories).
    pub fn query_sessions_detailed(
        &self,
        source: Option<SourceFilter>,
        project: Option<&str>,
        cwd: Option<&str>,
        since_ms: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<SessionDetailRow>> {
        self.query_sessions_detailed_filtered(source, project, cwd, since_ms, None, limit)
    }

    pub fn query_sessions_detailed_filtered(
        &self,
        source: Option<SourceFilter>,
        project: Option<&str>,
        cwd: Option<&str>,
        since_ms: Option<u64>,
        kind: Option<SessionKindFilter>,
        limit: Option<usize>,
    ) -> Result<Vec<SessionDetailRow>> {
        self.query_sessions_detailed_selected(
            source, project, cwd, since_ms, kind, None, None, limit,
        )
    }

    /// Apply exact identity selectors together with all listing filters before limiting rows.
    #[allow(clippy::too_many_arguments)]
    pub fn query_sessions_detailed_selected(
        &self,
        source: Option<SourceFilter>,
        project: Option<&str>,
        cwd: Option<&str>,
        since_ms: Option<u64>,
        kind: Option<SessionKindFilter>,
        session_id: Option<&str>,
        source_path: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<SessionDetailRow>> {
        let mut sql = String::from(
            "SELECT source, session_id, source_path, project, repo_project,
                    cwd, git_root, started_at, last_at, message_count, label, conversation_kind
             FROM sessions",
        );
        let (predicate, mut values) = session_selection_sql(
            source,
            project,
            cwd,
            since_ms,
            kind,
            session_id,
            source_path,
        );
        sql.push_str(&predicate);
        sql.push_str(" ORDER BY last_at DESC");
        if let Some(limit) = limit {
            sql.push_str(" LIMIT ?");
            values.push(rusqlite::types::Value::Integer(limit as i64));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            let source_label: String = row.get(0)?;
            let source = SourceKind::from_label(&source_label).unwrap_or(SourceKind::Claude);
            let repo_project: Option<String> = row.get(4)?;
            Ok(SessionDetailRow {
                source,
                session_id: row.get(1)?,
                source_path: row.get(2)?,
                project: row.get(3)?,
                repo_project: repo_project.filter(|value| !value.is_empty()),
                cwd: row.get::<_, Option<String>>(5)?.filter(|v| !v.is_empty()),
                git_root: row.get::<_, Option<String>>(6)?.filter(|v| !v.is_empty()),
                started_at: row.get::<_, i64>(7)?.max(0) as u64,
                last_at: row.get::<_, i64>(8)?.max(0) as u64,
                message_count: row.get::<_, i64>(9)?.max(0) as u64,
                label: row.get::<_, Option<String>>(10)?.filter(|v| !v.is_empty()),
                conversation_kind: row.get::<_, Option<String>>(11)?.filter(|v| !v.is_empty()),
            })
        })?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Count exactly the same identities and predicates as session listing, without detail rows.
    #[allow(clippy::too_many_arguments)]
    pub fn count_sessions_selected(
        &self,
        source: Option<SourceFilter>,
        project: Option<&str>,
        cwd: Option<&str>,
        since_ms: Option<u64>,
        kind: Option<SessionKindFilter>,
        session_id: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<u64> {
        let (predicate, values) = session_selection_sql(
            source,
            project,
            cwd,
            since_ms,
            kind,
            session_id,
            source_path,
        );
        Ok(self.conn.query_row(
            &format!("SELECT count(*) FROM sessions{predicate}"),
            params_from_iter(values),
            |row| row.get(0),
        )?)
    }

    /// Count indexed identities using canonical whole-session origin metadata.
    /// A missing row is distinct from a known primary session with a NULL kind.
    pub fn count_session_scopes(
        &self,
        scopes: &HashSet<(SourceKind, String, String)>,
        kind: SessionKindFilter,
    ) -> Result<Option<u64>> {
        if kind == SessionKindFilter::All {
            return Ok(Some(scopes.len() as u64));
        }
        let mut stmt = self.conn.prepare("SELECT conversation_kind FROM sessions WHERE source = ?1 AND session_id = ?2 AND source_path = ?3")?;
        let mut count = 0;
        for (source, session_id, source_path) in scopes {
            let metadata = stmt
                .query_row(
                    params![source.storage_label(), session_id, source_path],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?;
            let Some(metadata) = metadata else {
                return Ok(None);
            };
            if kind.matches_kind(metadata.as_deref().filter(|value| !value.is_empty())) {
                count += 1;
            }
        }
        Ok(Some(count))
    }

    /// Stored conversation kind for one exact session identity, if the
    /// analytics cache has a row for it. Search-result grouping uses this
    /// complete-session truth so a session is never classified from only
    /// the matched records; missing rows yield `None` and callers fall
    /// back to hit-derived kinds.
    pub fn session_conversation_kind(
        &self,
        source: &str,
        session_id: &str,
        source_path: &str,
    ) -> Option<String> {
        self.conn
            .query_row(
                "SELECT conversation_kind FROM sessions
                 WHERE source = ?1 AND session_id = ?2 AND source_path = ?3",
                params![source, session_id, source_path],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()
            .ok()
            .flatten()
            .flatten()
            .filter(|kind| !kind.is_empty())
    }

    /// Aggregate in SQLite rather than materializing every session in the client.
    /// One stored row is one (source, session_id, source_path) session identity.
    /// Match repository grouping and the default regular-session filter.
    pub fn query_project_summaries(
        &self,
        source: Option<SourceFilter>,
    ) -> Result<Vec<ProjectSummary>> {
        let mut sql = format!(
            "with project_sessions as (
                select {REPOSITORY_PROJECT_SQL} as project,
                       last_at
                from sessions
                where (conversation_kind is null or conversation_kind != 'guardian_review')",
        );
        let mut values: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(source) = source {
            let labels = source.storage_labels();
            let placeholders = std::iter::repeat_n("?", labels.len())
                .collect::<Vec<_>>()
                .join(", ");
            sql.push_str(&format!(" and source in ({placeholders})"));
            values.extend(
                labels
                    .iter()
                    .map(|label| rusqlite::types::Value::Text((*label).to_string())),
            );
        }
        sql.push_str(
            ")
             select project, count(*) as session_count,
                    max(case when last_at > 0 then last_at end) as last_at
             from project_sessions
             group by project
             order by last_at desc, project asc",
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            Ok(ProjectSummary {
                project: row.get(0)?,
                session_count: row.get(1)?,
                last_at: row.get(2)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn query_projects(
        &self,
        source: Option<SourceFilter>,
        grouping: ProjectGrouping,
    ) -> Result<Vec<String>> {
        let project_expr = match grouping {
            ProjectGrouping::Flat => "project",
            ProjectGrouping::Repository => REPOSITORY_PROJECT_SQL,
        };
        let mut sql = format!("SELECT DISTINCT {project_expr} FROM sessions");
        let mut values: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(source) = source {
            let labels = source.storage_labels();
            let placeholders = std::iter::repeat_n("?", labels.len())
                .collect::<Vec<_>>()
                .join(", ");
            sql.push_str(&format!(" WHERE source IN ({placeholders})"));
            values.extend(
                labels
                    .iter()
                    .map(|label| rusqlite::types::Value::Text((*label).to_string())),
            );
        }
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| row.get::<_, String>(0))?;
        let mut projects = Vec::new();
        for row in rows {
            let project = display_project_name(&row?);
            if !project.is_empty() {
                projects.push(project);
            }
        }
        projects.sort_by(|left, right| {
            match (
                left.as_str() == UNFILED_PROJECT,
                right.as_str() == UNFILED_PROJECT,
            ) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => left.cmp(right),
            }
        });
        projects.dedup();
        Ok(projects)
    }

    pub fn query_source_timestamps(&self, since_ms: Option<u64>) -> Result<Vec<(SourceKind, u64)>> {
        self.query_source_timestamps_filtered(
            None,
            since_ms,
            None,
            None,
            ProjectGrouping::Flat,
            None,
        )
    }

    pub fn query_source_timestamps_filtered(
        &self,
        source: Option<SourceFilter>,
        since_ms: Option<u64>,
        until_ms: Option<u64>,
        project: Option<&str>,
        grouping: ProjectGrouping,
        kind: Option<SessionKindFilter>,
    ) -> Result<Vec<(SourceKind, u64)>> {
        let mut sql = String::from("SELECT source, last_at FROM sessions");
        let mut clauses = Vec::new();
        let mut values: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(source) = source {
            let labels = source.storage_labels();
            let placeholders = std::iter::repeat_n("?", labels.len())
                .collect::<Vec<_>>()
                .join(", ");
            clauses.push(format!("source IN ({placeholders})"));
            values.extend(
                labels
                    .iter()
                    .map(|label| rusqlite::types::Value::Text((*label).to_string())),
            );
        }
        if let Some(since_ms) = since_ms {
            clauses.push("last_at >= ?".to_string());
            values.push(rusqlite::types::Value::Integer(since_ms as i64));
        }
        if let Some(until_ms) = until_ms {
            clauses.push("last_at <= ?".to_string());
            values.push(rusqlite::types::Value::Integer(until_ms as i64));
        }
        if let Some(project) = project {
            let project_expr = match grouping {
                ProjectGrouping::Flat => "project",
                ProjectGrouping::Repository => REPOSITORY_PROJECT_SQL,
            };
            clauses.push(format!("{project_expr} = ?"));
            values.push(rusqlite::types::Value::Text(project.to_string()));
        }
        if let Some(kind) = kind {
            match kind {
                SessionKindFilter::Primary => {
                    clauses.push(
                        "(conversation_kind IS NULL OR conversation_kind = 'main')".to_string(),
                    );
                }
                SessionKindFilter::Subagent => {
                    clauses.push(
                        "conversation_kind IS NOT NULL AND conversation_kind NOT IN ('main', 'guardian_review')".to_string(),
                    );
                }
                SessionKindFilter::Regular => {
                    clauses.push(
                        "(conversation_kind IS NULL OR conversation_kind != 'guardian_review')"
                            .to_string(),
                    );
                }
                SessionKindFilter::All => {}
            }
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?.max(0) as u64,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (label, ts) = row?;
            if let Some(kind) = SourceKind::from_label(&label) {
                out.push((kind, ts));
            }
        }
        Ok(out)
    }

    pub fn query_source_labels(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT DISTINCT source FROM sessions")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        out.sort();
        Ok(out)
    }

    pub fn query_project_timestamps(
        &self,
        source: Option<SourceFilter>,
        since_ms: Option<u64>,
        grouping: ProjectGrouping,
    ) -> Result<Vec<(String, u64)>> {
        let project_expr = match grouping {
            ProjectGrouping::Flat => "project",
            ProjectGrouping::Repository => REPOSITORY_PROJECT_SQL,
        };
        let mut sql = format!("SELECT {project_expr}, last_at FROM sessions");
        let mut clauses = Vec::new();
        let mut values: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(source) = source {
            let labels = source.storage_labels();
            let placeholders = std::iter::repeat_n("?", labels.len())
                .collect::<Vec<_>>()
                .join(", ");
            clauses.push(format!("source IN ({placeholders})"));
            values.extend(
                labels
                    .iter()
                    .map(|label| rusqlite::types::Value::Text((*label).to_string())),
            );
        }
        if let Some(since_ms) = since_ms {
            clauses.push("last_at >= ?".to_string());
            values.push(rusqlite::types::Value::Integer(since_ms as i64));
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?.max(0) as u64,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (project, last_at) = row?;
            out.push((display_project_name(&project), last_at));
        }
        Ok(out)
    }

    pub fn project_for_session(
        &self,
        source: SourceKind,
        session_id: &str,
        source_path: &str,
        grouping: ProjectGrouping,
    ) -> Result<Option<String>> {
        let display_expr = match grouping {
            ProjectGrouping::Flat => "project",
            ProjectGrouping::Repository => REPOSITORY_PROJECT_SQL,
        };
        let project: Option<String> = self
            .conn
            .query_row(
                &format!(
                    "SELECT {display_expr} FROM sessions
                     WHERE source = ?1 AND session_id = ?2 AND source_path = ?3"
                ),
                params![source.storage_label(), session_id, source_path],
                |row| row.get(0),
            )
            .optional()?;
        Ok(project.map(|project| display_project_name(&project)))
    }

    pub fn raw_projects_for_repository(&self, repo: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT project FROM sessions
             WHERE (repo_project = ?1 OR (repo_project IS NULL AND project = ?1))
               AND project IS NOT NULL AND project != ''",
        )?;
        let rows = stmt.query_map([repo], |row| row.get::<_, Option<String>>(0))?;
        let mut out = Vec::new();
        for row in rows {
            if let Some(proj) = row?.filter(|s| !s.is_empty()) {
                out.push(proj);
            }
        }
        Ok(out)
    }

    pub fn query_session_projects(
        &self,
        sessions: &[(SourceKind, String, String)],
        grouping: ProjectGrouping,
    ) -> Result<HashMap<(SourceKind, String, String), String>> {
        if sessions.is_empty() {
            return Ok(HashMap::new());
        }
        let display_expr = match grouping {
            ProjectGrouping::Flat => "project",
            ProjectGrouping::Repository => REPOSITORY_PROJECT_SQL,
        };
        let conditions = std::iter::repeat_n(
            "(source = ? AND session_id = ? AND source_path = ?)",
            sessions.len(),
        )
        .collect::<Vec<_>>()
        .join(" OR ");
        let mut stmt = self.conn.prepare(&format!(
            "SELECT source, session_id, source_path, {display_expr}
             FROM sessions WHERE {conditions}"
        ))?;
        let values = sessions
            .iter()
            .flat_map(|(source, session_id, source_path)| {
                [
                    rusqlite::types::Value::Text(source.storage_label().to_string()),
                    rusqlite::types::Value::Text(session_id.clone()),
                    rusqlite::types::Value::Text(source_path.clone()),
                ]
            });
        let rows = stmt.query_map(params_from_iter(values), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut projects = HashMap::new();
        for row in rows {
            let (source, session_id, source_path, project) = row?;
            let Some(source) = SourceKind::from_label(&source) else {
                continue;
            };
            projects.insert(
                (source, session_id, source_path),
                display_project_name(&project),
            );
        }
        Ok(projects)
    }

    pub fn query_session_cwd(
        &self,
        source: SourceKind,
        session_id: &str,
        source_path: Option<&str>,
    ) -> Result<Option<String>> {
        if let Some(path) = source_path {
            let cwd: Option<Option<String>> = self
                .conn
                .query_row(
                    "SELECT COALESCE(NULLIF(cwd, ''), NULLIF(git_root, ''))
                     FROM sessions
                     WHERE source = ?1 AND session_id = ?2 AND source_path = ?3",
                    params![source.storage_label(), session_id, path],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?;
            return Ok(cwd.flatten().filter(|s| !s.is_empty()));
        }
        // If source_path was not provided, query all distinct non-empty cwds for this (source, session_id).
        // If exactly 1 distinct cwd exists, return it. If multiple distinct cwds exist,
        // it is ambiguous across multiple worktrees/paths, so reject.
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT COALESCE(NULLIF(cwd, ''), NULLIF(git_root, '')) AS effective_cwd
             FROM sessions
             WHERE source = ?1 AND session_id = ?2
               AND ((cwd IS NOT NULL AND cwd != '') OR (git_root IS NOT NULL AND git_root != ''))",
        )?;
        let mut rows = stmt.query(params![source.storage_label(), session_id])?;
        let mut found_cwd: Option<String> = None;
        while let Some(row) = rows.next()? {
            let cwd: Option<String> = row.get(0)?;
            if let Some(c) = cwd.filter(|s| !s.is_empty()) {
                if found_cwd.is_some() {
                    // Ambiguous: multiple distinct cwds for this session_id across different paths!
                    return Ok(None);
                }
                found_cwd = Some(c);
            }
        }
        Ok(found_cwd)
    }

    pub fn query_session_cwds(
        &self,
        sessions: &[(SourceKind, String, String)],
    ) -> Result<HashMap<(SourceKind, String, String), String>> {
        if sessions.is_empty() {
            return Ok(HashMap::new());
        }
        let mut cwds = HashMap::new();
        // Chunk in batches of 100 to stay well under SQLite's 999 parameter limit
        for chunk in sessions.chunks(100) {
            let conditions = std::iter::repeat_n(
                "(source = ? AND session_id = ? AND source_path = ?)",
                chunk.len(),
            )
            .collect::<Vec<_>>()
            .join(" OR ");
            let mut stmt = self.conn.prepare(&format!(
                "SELECT source, session_id, source_path, COALESCE(NULLIF(cwd, ''), NULLIF(git_root, ''))
                 FROM sessions WHERE {conditions}"
            ))?;
            let values = chunk.iter().flat_map(|(source, session_id, source_path)| {
                [
                    rusqlite::types::Value::Text(source.storage_label().to_string()),
                    rusqlite::types::Value::Text(session_id.clone()),
                    rusqlite::types::Value::Text(source_path.clone()),
                ]
            });
            let rows = stmt.query_map(params_from_iter(values), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            })?;
            for row in rows {
                let (source, session_id, source_path, cwd) = row?;
                let Some(source) = SourceKind::from_label(&source) else {
                    continue;
                };
                if let Some(cwd) = cwd.filter(|s| !s.is_empty()) {
                    cwds.insert((source, session_id, source_path), cwd);
                }
            }
        }
        Ok(cwds)
    }
}

impl AnalyticsWriter {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            store: AnalyticsStore::open(path)?,
            sessions: HashMap::new(),
            metadata_cache: HashMap::new(),
            git_cache: HashMap::new(),
            cwd_overrides: HashMap::new(),
            opencode_cache: OpencodeLookupCache::default(),
        })
    }

    pub fn clear(&self) -> Result<()> {
        self.store.clear()
    }

    pub fn delete_source_path(&self, source_path: &str) -> Result<()> {
        self.store.delete_source_path(source_path)
    }

    pub fn delete_session_scope(&self, scope: &SessionScope) -> Result<()> {
        self.store.delete_session_scope(scope)
    }

    pub fn set_session_cwd(
        &mut self,
        source: SourceKind,
        source_path: &str,
        session_id: &str,
        cwd: &str,
    ) {
        self.cwd_overrides.insert(
            SessionKey {
                source,
                session_id: session_id.to_string(),
                source_path: source_path.to_string(),
            },
            cwd.to_string(),
        );
    }

    pub fn record(&mut self, record: &Record) -> Result<()> {
        let key = SessionKey {
            source: record.source,
            session_id: record.session_id.clone(),
            source_path: record.source_path.clone(),
        };
        let entry = self
            .sessions
            .entry(key.clone())
            .or_insert_with(|| SessionAccumulator {
                key,
                project: record.project.clone(),
                started_at: record.ts,
                last_at: record.ts,
                message_count: 0,
                first_user_text: None,
                conversation_kind: None,
            });
        if record.ts < entry.started_at {
            entry.started_at = record.ts;
        }
        if record.ts >= entry.last_at {
            entry.last_at = record.ts;
            if !record.project.is_empty() {
                entry.project = record.project.clone();
            }
        }
        entry.message_count = entry.message_count.saturating_add(1);
        if entry.first_user_text.is_none()
            && record.role == "user"
            && !record.text.trim().is_empty()
            && !sanitize_label(&record.text).is_empty()
        {
            entry.first_user_text = Some(record.text.clone());
        }
        // Prefer an explicit "main" over per-record non-main kinds: Pi and
        // OpenClaw stamp compaction/branch on entries inside otherwise-main
        // sessions, and Claude stamps sidechain lines the same way. A session
        // is non-interactive only when no record claims it as main.
        if let Some(kind) = record.links.conversation_kind.clone()
            && !kind.is_empty()
            && (entry.conversation_kind.is_none() || kind == "main")
        {
            entry.conversation_kind = Some(kind);
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.sessions.is_empty() {
            return Ok(());
        }
        let pending_sessions: Vec<SessionAccumulator> = self.sessions.values().cloned().collect();
        let sessions: Vec<(SessionAccumulator, SessionMetadata)> = pending_sessions
            .into_iter()
            .map(|session| {
                let metadata = self.resolve_metadata(&session.key);
                (session, metadata)
            })
            .collect();
        let tx = self.store.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO sessions(
                    source, session_id, source_path, project, cwd, git_root, git_common_dir,
                    repo_project, started_at, last_at, message_count, resolution_status,
                    label, conversation_kind
                )
                VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                ON CONFLICT(source, session_id, source_path) DO UPDATE SET
                    project = excluded.project,
                    cwd = excluded.cwd,
                    git_root = excluded.git_root,
                    git_common_dir = excluded.git_common_dir,
                    repo_project = excluded.repo_project,
                    started_at = MIN(sessions.started_at, excluded.started_at),
                    last_at = MAX(sessions.last_at, excluded.last_at),
                    message_count = sessions.message_count + excluded.message_count,
                    resolution_status = excluded.resolution_status,
                    -- The stored label/kind describe the session's opening and
                    -- survive incremental deltas, which only see mid-session
                    -- records. Corrections flow through parser-version bumps,
                    -- which delete the row first (delete_first) and recompute.
                    label = COALESCE(sessions.label, excluded.label),
                    conversation_kind = COALESCE(sessions.conversation_kind, excluded.conversation_kind)
                "#,
            )?;
            // OpenCode title/agent/cwd lookups hit the source SQLite database.
            // Sessions from one database share the same file, so memoize in
            // writer's opencode_cache to avoid reopening it repeatedly.
            for (session, metadata) in sessions {
                let label = extract_session_label(
                    session.key.source,
                    &session.key.source_path,
                    &session.key.session_id,
                    session.first_user_text.as_deref(),
                    metadata.cwd.as_deref(),
                    &mut self.opencode_cache,
                );
                let conversation_kind = infer_session_kind(
                    session.key.source,
                    &session.key.source_path,
                    &session.key.session_id,
                    session.conversation_kind.as_deref(),
                    metadata.cwd.as_deref(),
                    session.first_user_text.as_deref(),
                    &mut self.opencode_cache,
                );
                stmt.execute(params![
                    session.key.source.storage_label(),
                    session.key.session_id,
                    session.key.source_path,
                    session.project,
                    metadata.cwd,
                    metadata.git_root,
                    metadata.git_common_dir,
                    metadata.repo_project,
                    session.started_at as i64,
                    session.last_at as i64,
                    session.message_count as i64,
                    metadata.resolution_status,
                    label,
                    conversation_kind,
                ])?;
            }
        }
        tx.commit()?;
        self.sessions.clear();
        Ok(())
    }

    fn resolve_metadata(&mut self, key: &SessionKey) -> SessionMetadata {
        if let Some(cached) = self.metadata_cache.get(key) {
            return cached.clone();
        }
        let metadata = self.resolve_uncached_metadata(key);
        self.metadata_cache.insert(key.clone(), metadata.clone());
        metadata
    }

    fn resolve_uncached_metadata(&mut self, key: &SessionKey) -> SessionMetadata {
        let cwd = self.cwd_overrides.get(key).cloned().or_else(|| {
            resolve_session_cwd_from_parts(
                key.source,
                &key.source_path,
                &key.session_id,
                &mut self.opencode_cache,
            )
        });
        let Some(cwd) = cwd else {
            return SessionMetadata {
                resolution_status: "no-cwd".to_string(),
                ..SessionMetadata::default()
            };
        };
        let git = self
            .git_cache
            .entry(cwd.clone())
            .or_insert_with(|| git_metadata_for_cwd(&cwd))
            .clone();
        SessionMetadata {
            cwd: Some(cwd),
            git_root: git.git_root,
            git_common_dir: git.git_common_dir,
            repo_project: git.repo_project,
            resolution_status: git.status,
        }
    }
}

#[derive(Clone, Default)]
struct GitMetadata {
    git_root: Option<String>,
    git_common_dir: Option<String>,
    repo_project: Option<String>,
    status: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorktreeRepoInfo {
    repo_project: String,
    git_root: Option<String>,
    git_common_dir: Option<String>,
}

fn git_metadata_for_cwd(cwd: &str) -> GitMetadata {
    let deadline = Instant::now() + GIT_METADATA_TIMEOUT;
    let mut root = git_rev_parse(cwd, &["rev-parse", "--show-toplevel"], deadline);
    let mut common_dir = git_rev_parse(
        cwd,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        deadline,
    );
    let fallback = worktree_repo_project(cwd);
    let path_repo_project =
        claude_worktree_repo_project(cwd).or_else(|| codex_worktree_repo_project(cwd));
    let mut repo_project = common_dir
        .as_deref()
        .and_then(common_dir_project_name)
        .or_else(|| root.as_deref().and_then(path_file_name))
        .or(path_repo_project);

    let mut is_fallback = false;
    if let Some(fb) = fallback
        && (common_dir.is_none() || repo_project.is_none())
    {
        repo_project = Some(fb.repo_project);
        if root.is_none() {
            root = fb.git_root;
        }
        if common_dir.is_none() {
            common_dir = fb.git_common_dir;
        }
        is_fallback = true;
    }

    let status = if is_fallback {
        "path-fallback"
    } else if repo_project.is_some() {
        "ok"
    } else if root.is_some() || common_dir.is_some() {
        "git-partial"
    } else {
        "not-git"
    }
    .to_string();

    GitMetadata {
        git_root: root,
        git_common_dir: common_dir,
        repo_project,
        status,
    }
}

pub(crate) fn repository_project_for_cwd(cwd: &str) -> Option<String> {
    git_metadata_for_cwd(cwd).repo_project
}

fn get_development_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        let home_path = PathBuf::from(home);
        for sub in [
            "Developer",
            "Development",
            "Code",
            "Projects",
            "src",
            "repos",
            "workspace",
            "workspaces",
        ] {
            let candidate = home_path.join(sub);
            if candidate.is_dir() {
                roots.push(candidate);
            }
        }
        roots.push(home_path);
    }
    roots
}

fn decode_claude_encoded_path(name: &str) -> Option<PathBuf> {
    let name = if let Some(idx) = name.find("-Users-") {
        &name[idx + 1..]
    } else if let Some(idx) = name.find("-home-") {
        &name[idx + 1..]
    } else {
        name.strip_prefix('-')?
    };
    if !(name.starts_with("Users-") || name.starts_with("home-")) {
        return None;
    }
    let parts: Vec<&str> = name.split('-').collect();
    let root = if name.starts_with("Users-") {
        PathBuf::from("/Users")
    } else {
        PathBuf::from("/home")
    };
    let mut curr = root;
    let mut idx = 1;
    while idx < parts.len() {
        let mut found = false;
        for end in (idx + 1..=parts.len()).rev() {
            let segment = parts[idx..end].join("-");
            let candidate = curr.join(&segment);
            if candidate.is_dir() {
                curr = candidate;
                idx = end;
                found = true;
                break;
            }
        }
        if !found {
            break;
        }
    }
    if curr.join(".git").exists() || curr.join("HEAD").exists() {
        Some(curr)
    } else {
        None
    }
}

fn claude_cwd_from_source_path(source_path: &str) -> Option<String> {
    let path = Path::new(source_path);
    let folder = path.parent()?.file_name()?.to_str()?;
    if (folder.contains("-Users-") || folder.contains("-home-"))
        && let Some(repo_dir) = decode_claude_encoded_path(folder)
    {
        return Some(repo_dir.to_string_lossy().to_string());
    }
    if let Some(rest) = folder.strip_prefix("-private-tmp-") {
        return Some(format!("/private/tmp/{rest}"));
    }
    if let Some(rest) = folder.strip_prefix("-tmp-") {
        return Some(format!("/tmp/{rest}"));
    }
    None
}

fn worktree_repo_project(cwd: &str) -> Option<WorktreeRepoInfo> {
    let path = Path::new(cwd);

    // 1. Direct .git file check (authoritative when worktree directory exists on disk)
    // Submodule protection: worktree gitdirs have a "commondir" file pointing to the main repo's git dir.
    let git_file = path.join(".git");
    if git_file.is_file()
        && let Ok(content) = std::fs::read_to_string(&git_file)
        && let Some(line) = content
            .lines()
            .find(|l| l.trim_start().starts_with("gitdir:"))
    {
        let gitdir_str = line.trim_start()["gitdir:".len()..].trim();
        let gitdir_path = path.join(gitdir_str);
        let commondir_file = gitdir_path.join("commondir");
        if commondir_file.is_file()
            && let Ok(common_str) = std::fs::read_to_string(&commondir_file)
        {
            let common_dir = gitdir_path.join(common_str.trim());
            if let Ok(canonical_common) = common_dir.canonicalize() {
                let repo_project =
                    common_dir_project_name(canonical_common.to_string_lossy().as_ref())?;
                let git_root =
                    if canonical_common.file_name().and_then(|n| n.to_str()) == Some(".git") {
                        canonical_common
                            .parent()
                            .map(|p| p.to_string_lossy().to_string())
                    } else {
                        canonical_common
                            .parent()
                            .map(|p| p.to_string_lossy().to_string())
                    };
                return Some(WorktreeRepoInfo {
                    repo_project,
                    git_root,
                    git_common_dir: Some(canonical_common.to_string_lossy().to_string()),
                });
            }
        }
        // Has a .git file pointing to gitdir, but lacks commondir: this is a submodule, not a worktree.
        // Do not fall through to name heuristics.
        return None;
    }

    // 2. Ancestor worktree paths (.claude/worktrees/*, .worktrees/*, or worktrees/*)
    for ancestor in path.ancestors() {
        let name = ancestor.file_name().and_then(|n| n.to_str());
        if name == Some("worktrees") || name == Some(".worktrees") {
            let Some(parent) = ancestor.parent() else {
                continue;
            };
            if parent.file_name().and_then(|n| n.to_str()) == Some(".git") {
                continue;
            }
            let parent_name = parent.file_name().and_then(|n| n.to_str());
            if parent_name == Some(".codex") || parent_name == Some(".grok") {
                if let Some(leaf) = path.file_name().and_then(|n| n.to_str()) {
                    for dev_root in get_development_roots() {
                        let candidate = dev_root.join(leaf);
                        let candidate_git = candidate.join(".git");
                        if candidate_git.exists() {
                            let git_root = candidate.to_string_lossy().to_string();
                            let git_common_dir = candidate_git.to_string_lossy().to_string();
                            let actual_name = candidate
                                .canonicalize()
                                .ok()
                                .and_then(|p| {
                                    p.file_name().map(|n| n.to_string_lossy().to_string())
                                })
                                .unwrap_or_else(|| leaf.to_string());
                            return Some(WorktreeRepoInfo {
                                repo_project: actual_name,
                                git_root: Some(git_root),
                                git_common_dir: Some(git_common_dir),
                            });
                        }
                    }
                }
                continue;
            }
            let is_claude = parent_name == Some(".claude");
            let repo_dir = if is_claude {
                let Some(p) = parent.parent() else {
                    continue;
                };
                p
            } else {
                parent
            };
            let common_dir = repo_dir.join(".git");
            if !is_claude && !common_dir.exists() && !repo_dir.join("HEAD").exists() {
                continue;
            }
            let Some(repo_name) = path_file_name(repo_dir.to_string_lossy().as_ref()) else {
                continue;
            };
            let git_root = repo_dir.to_string_lossy().to_string();
            let git_common_dir = if common_dir.exists() {
                Some(common_dir.to_string_lossy().to_string())
            } else {
                Some(git_root.clone())
            };
            return Some(WorktreeRepoInfo {
                repo_project: repo_name,
                git_root: Some(git_root),
                git_common_dir,
            });
        } else if let Some(name_str) = name {
            if let Some(repo_name) = name_str
                .strip_suffix("-worktrees")
                .or_else(|| name_str.strip_suffix(".worktrees"))
            {
                if let Some(parent) = ancestor.parent() {
                    let candidate = parent.join(repo_name);
                    let common_dir = candidate.join(".git");
                    if common_dir.exists() || candidate.join("HEAD").exists() {
                        let git_root = candidate.to_string_lossy().to_string();
                        let git_common_dir = if common_dir.exists() {
                            Some(common_dir.to_string_lossy().to_string())
                        } else {
                            Some(git_root.clone())
                        };
                        let actual_name = candidate
                            .canonicalize()
                            .ok()
                            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                            .unwrap_or_else(|| repo_name.to_string());
                        return Some(WorktreeRepoInfo {
                            repo_project: actual_name,
                            git_root: Some(git_root),
                            git_common_dir,
                        });
                    }
                }
            } else if (name_str.contains("-Users-") || name_str.contains("-home-"))
                && let Some(repo_dir) = decode_claude_encoded_path(name_str)
                && let Some(repo_name) = path_file_name(repo_dir.to_string_lossy().as_ref())
            {
                let common_dir = repo_dir.join(".git");
                let git_root = repo_dir.to_string_lossy().to_string();
                let git_common_dir = if common_dir.exists() {
                    Some(common_dir.to_string_lossy().to_string())
                } else {
                    Some(git_root.clone())
                };
                return Some(WorktreeRepoInfo {
                    repo_project: repo_name,
                    git_root: Some(git_root),
                    git_common_dir,
                });
            }
        }
    }

    // 3. Sibling worktree naming convention (handles deleted worktrees)
    // Common delimiters: .wt-, -wt-, .worktree-, -worktree-, or any dot prefix
    let leaf = path.file_name().and_then(|n| n.to_str())?;
    for delimiter in [".wt-", "-wt-", ".worktree-", "-worktree-"] {
        if let Some((repo_prefix, _)) = leaf.rsplit_once(delimiter)
            && !repo_prefix.is_empty()
            && let Some(parent) = path.parent()
        {
            let candidate = parent.join(repo_prefix);
            let candidate_git = candidate.join(".git");
            if candidate_git.exists() {
                let git_root = candidate.to_string_lossy().to_string();
                let git_common_dir = candidate_git.to_string_lossy().to_string();
                let actual_name = candidate
                    .canonicalize()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                    .unwrap_or_else(|| repo_prefix.to_string());
                return Some(WorktreeRepoInfo {
                    repo_project: actual_name,
                    git_root: Some(git_root),
                    git_common_dir: Some(git_common_dir),
                });
            }
        }
    }

    if let Some((repo_prefix, _)) = leaf.split_once('.')
        && !repo_prefix.is_empty()
        && let Some(parent) = path.parent()
    {
        let candidate = parent.join(repo_prefix);
        let candidate_git = candidate.join(".git");
        if candidate != path && candidate_git.exists() {
            let git_root = candidate.to_string_lossy().to_string();
            let git_common_dir = candidate_git.to_string_lossy().to_string();
            let actual_name = candidate
                .canonicalize()
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                .unwrap_or_else(|| repo_prefix.to_string());
            return Some(WorktreeRepoInfo {
                repo_project: actual_name,
                git_root: Some(git_root),
                git_common_dir: Some(git_common_dir),
            });
        }
    }

    // 4. Temporary/worker directory prefix check (handles agent workers under /tmp, /private/tmp, etc.)
    let path_str = path.to_string_lossy();
    if path_str.starts_with("/tmp")
        || path_str.starts_with("/private/tmp")
        || path_str.starts_with("/var/folders")
        || path_str.starts_with("/private/var/folders")
    {
        let mut temp_component = None;
        let components: Vec<_> = path.components().collect();
        for (i, comp) in components.iter().enumerate() {
            let s = comp.as_os_str().to_string_lossy();
            if (s == "tmp" || s == "T") && i + 1 < components.len() {
                temp_component = Some(components[i + 1].as_os_str().to_string_lossy().to_string());
                break;
            }
        }
        if let Some(comp) = temp_component {
            let mut prefixes = Vec::new();
            for delimiter in [
                "-worker", ".worker", "-sandbox", ".sandbox", "-wt-", ".wt-", "-task-", ".task-",
            ] {
                if let Some((p, _)) = comp.split_once(delimiter) {
                    prefixes.push(p);
                }
            }
            if let Some((p, _)) = comp.rsplit_once('-') {
                prefixes.push(p);
            }
            if let Some((p, _)) = comp.split_once('.') {
                prefixes.push(p);
            }
            if let Some((p, _)) = comp.split_once('-') {
                prefixes.push(p);
            }
            prefixes.push(&comp);
            let dev_roots = get_development_roots();
            for prefix in prefixes {
                if prefix.is_empty() {
                    continue;
                }
                for root in &dev_roots {
                    let candidate = root.join(prefix);
                    let candidate_git = candidate.join(".git");
                    if candidate_git.exists() {
                        let git_root = candidate.to_string_lossy().to_string();
                        let git_common_dir = candidate_git.to_string_lossy().to_string();
                        let actual_name = candidate
                            .canonicalize()
                            .ok()
                            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                            .unwrap_or_else(|| prefix.to_string());
                        return Some(WorktreeRepoInfo {
                            repo_project: actual_name,
                            git_root: Some(git_root),
                            git_common_dir: Some(git_common_dir),
                        });
                    }
                }
            }
        }
    }

    None
}

fn codex_worktree_repo_project(cwd: &str) -> Option<String> {
    let cwd = Path::new(cwd);
    for ancestor in cwd.ancestors() {
        if ancestor.file_name().and_then(|name| name.to_str()) != Some("worktrees") {
            continue;
        }
        let codex_dir = ancestor.parent()?;
        if codex_dir.file_name().and_then(|name| name.to_str()) != Some(".codex") {
            continue;
        }
        let mut relative = cwd.strip_prefix(ancestor).ok()?.components();
        relative.next()?;
        return relative
            .next()
            .and_then(|component| component.as_os_str().to_str())
            .filter(|name| !name.is_empty())
            .map(str::to_string);
    }
    None
}

fn claude_worktree_repo_project(cwd: &str) -> Option<String> {
    worktree_repo_project(cwd).map(|info| info.repo_project)
}

fn git_rev_parse(cwd: &str, args: &[&str], deadline: Instant) -> Option<String> {
    if Instant::now() >= deadline {
        return None;
    }
    let child = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let output = child_output_before(child, deadline)?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn child_output_before(mut child: Child, deadline: Instant) -> Option<Output> {
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) => {}
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(remaining.min(Duration::from_millis(10)));
    }
}

fn common_dir_project_name(path: &str) -> Option<String> {
    let path = Path::new(path);
    if path.file_name().and_then(|n| n.to_str()) == Some(".git") {
        return path
            .parent()
            .and_then(|p| path_file_name(p.to_string_lossy().as_ref()));
    }
    path_file_name(path.to_string_lossy().as_ref())
}

fn display_project_name(project: &str) -> String {
    decode_encoded_project_path(project).unwrap_or_else(|| project.to_string())
}

fn decode_encoded_project_path(project: &str) -> Option<String> {
    let trimmed = project.trim_matches('-');
    let lower = trimmed.to_lowercase();
    let home = std::env::var("HOME").ok();
    let home_user = home
        .as_deref()
        .and_then(|h| Path::new(h).file_name().and_then(|n| n.to_str()));
    let starts_with_user = home_user
        .map(|u| lower.starts_with(&format!("{}-", u.to_lowercase())))
        .unwrap_or(false);

    if !(lower.starts_with("users-")
        || lower.starts_with("home-")
        || lower.contains("-users-")
        || starts_with_user)
    {
        return None;
    }
    let parts: Vec<&str> = trimmed.split('-').filter(|part| !part.is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }

    if let Some(user) = home_user
        && parts[0].eq_ignore_ascii_case(user)
    {
        let tail = parts.get(1..)?;
        if !tail.is_empty() {
            return Some(encoded_tail_display(tail));
        }
    }

    if let Some(home) = home_relative_encoded_path(&parts) {
        return Some(home);
    }

    if parts[0].eq_ignore_ascii_case("home") {
        let tail = parts.get(2..)?;
        if tail.is_empty() {
            return None;
        }
        return Some(encoded_tail_display(tail));
    }

    let users_idx = parts
        .iter()
        .position(|part| part.eq_ignore_ascii_case("Users"))?;
    let tail = parts.get(users_idx + 2..)?;
    if tail.is_empty() {
        return None;
    }
    Some(encoded_tail_display(tail))
}

fn home_relative_encoded_path(parts: &[&str]) -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let mut home_parts = Path::new(&home)
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .filter(|part| !part.is_empty());
    let home_parent = home_parts.next_back()?;
    let users_idx = parts
        .iter()
        .position(|part| part.eq_ignore_ascii_case("Users"))?;
    if parts.get(users_idx + 1)? != &home_parent {
        return None;
    }
    let tail = parts.get(users_idx + 2..)?;
    if tail.is_empty() {
        return None;
    }
    Some(encoded_tail_display(tail))
}

fn encoded_tail_display(tail: &[&str]) -> String {
    if tail.len() == 1 {
        return format!("~/{}", tail[0]);
    }
    let common_dirs = [
        "projects",
        "code",
        "repos",
        "src",
        "dev",
        "work",
        "documents",
        "developer",
        "development",
        "workspace",
        "workspaces",
    ];
    if common_dirs.contains(&tail[0].to_lowercase().as_str()) && tail.len() > 1 {
        return tail[1..].join("-");
    }
    tail.join("-")
}

fn path_file_name(path: &str) -> Option<String> {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|name| !name.is_empty())
        .map(|name| name.to_string())
}

pub fn resolve_session_cwd_standalone(
    source: SourceKind,
    source_path: &str,
    session_id: &str,
) -> Option<String> {
    resolve_session_cwd_from_parts(
        source,
        source_path,
        session_id,
        &mut OpencodeLookupCache::default(),
    )
}

fn resolve_session_cwd_from_parts(
    source: SourceKind,
    source_path: &str,
    session_id: &str,
    opencode: &mut OpencodeLookupCache,
) -> Option<String> {
    if source == SourceKind::Opencode && crate::sources::opencode::is_database_path(source_path) {
        return crate::sources::opencode::enumerate_sessions(Path::new(source_path))
            .ok()?
            .into_iter()
            .find(|session| session.id == session_id)
            .map(|session| session.directory);
    }
    if source == SourceKind::Copilot
        && let Some(cwd) = resolve_copilot_workspace_cwd(source_path)
    {
        return Some(cwd);
    }
    if source == SourceKind::Grok
        && let Some(cwd) = crate::sources::grok::session_cwd(Path::new(source_path))
    {
        return Some(cwd);
    }
    if source == SourceKind::Jcode
        && let Some(cwd) = crate::sources::jcode::cwd_from_jcode_session(Path::new(source_path))
    {
        return Some(cwd.to_string_lossy().to_string());
    }
    if source == SourceKind::Muse
        && let Some(cwd) = crate::sources::muse::cwd_from_muse_session(Path::new(source_path))
    {
        return Some(cwd.to_string_lossy().to_string());
    }
    if source == SourceKind::Cursor
        && let Some(cwd) = crate::transfer::cwd_from_cursor_session(Path::new(source_path))
    {
        return Some(cwd.to_string_lossy().to_string());
    }
    if source == SourceKind::Opencode
        && let Some(cwd) = opencode.cwd(source_path, session_id)
    {
        return Some(cwd);
    }
    let file = std::fs::File::open(source_path).ok()?;
    let reader = std::io::BufReader::new(file);
    let mut fallback: Option<String> = None;
    for line in std::io::BufRead::lines(reader).map_while(std::result::Result::ok) {
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let cwd = value
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        if fallback.is_none() {
            fallback = cwd.clone();
        }

        let id_val = if matches!(
            source,
            SourceKind::Pi | SourceKind::OpenClaw | SourceKind::Omp
        ) {
            value.get("id").and_then(|v| v.as_str())
        } else {
            None
        };
        let session_id_match = value
            .get("sessionId")
            .and_then(|v| v.as_str())
            .or_else(|| value.get("session_id").and_then(|v| v.as_str()))
            .or(id_val)
            .map(|s| s == session_id)
            .unwrap_or(false);

        if session_id_match && cwd.is_some() {
            return cwd;
        }

        if source == SourceKind::Codex
            && value.get("type").and_then(|v| v.as_str()) == Some("session_meta")
        {
            let payload_cwd = value
                .get("payload")
                .and_then(|v| v.get("cwd"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            if payload_cwd.is_some() {
                return payload_cwd;
            }
        }

        if matches!(
            source,
            SourceKind::Pi | SourceKind::OpenClaw | SourceKind::Omp
        ) && value.get("type").and_then(|v| v.as_str()) == Some("session")
        {
            let id_matches = value
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s == session_id)
                .unwrap_or(false);
            if (id_matches || session_id_match) && cwd.is_some() {
                return cwd;
            }
        }
    }
    fallback.or_else(|| {
        if source == SourceKind::Claude {
            claude_cwd_from_source_path(source_path)
        } else {
            None
        }
    })
}

#[derive(Default)]
struct CopilotWorkspaceCwd {
    cwd: Option<String>,
    git_root: Option<String>,
}

fn resolve_copilot_workspace_cwd(source_path: &str) -> Option<String> {
    let workspace_path = Path::new(source_path).parent()?.join("workspace.yaml");
    let contents = std::fs::read_to_string(workspace_path).ok()?;
    let workspace = parse_copilot_workspace_cwd(&contents);
    workspace.cwd.or(workspace.git_root)
}

fn parse_copilot_workspace_cwd(contents: &str) -> CopilotWorkspaceCwd {
    let mut workspace = CopilotWorkspaceCwd::default();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || line.chars().next().is_some_and(|c| c.is_whitespace())
        {
            continue;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "cwd" => workspace.cwd = Some(value),
            "gitRoot" | "git_root" => workspace.git_root = Some(value),
            _ => {}
        }
    }
    workspace
}

#[allow(clippy::while_let_loop)]
pub fn sanitize_label(raw: &str) -> String {
    // Comprehensive stripping of system wrappers (case-insensitive).
    // Fast path: neither the tag stripper nor the generic unwrap can match
    // without a '<', so ordinary prose skips the owned buffer entirely and
    // goes straight to the single-allocation finish pass. This keeps the
    // per-record emptiness check in `record` cheap during index scans.
    if raw.contains('<') {
        let mut current = raw.to_string();
        const DROP_TAGS: &[&str] = &[
            "system-reminder",
            "command-message",
            "command-name",
            "local-command-stdout",
            "local-command-caveat",
            "local-command-output",
            "instructions",
            "environment_context",
            "cwd",
            "approval_policy",
            "shell",
            "user_instructions",
            "recommended_plugins",
            "skill",
            "user_action",
            "context",
            "task-notification",
            "task-id",
            "tool-use-id",
            "subagent_notification",
            "turn_aborted",
            "current_date",
            "timezone",
            "epoch",
            "collaboration_mode",
            "apps_instructions",
            "permissions",
            "total_tokens",
        ];
        for tag in DROP_TAGS {
            let open = format!("<{tag}");
            let close = format!("</{tag}>");
            loop {
                let Some(start) = find_ascii_ci(&current, &open) else {
                    break;
                };
                let open_end = match current[start..].find('>') {
                    Some(p) => start + p + 1,
                    None => {
                        current.truncate(start);
                        break;
                    }
                };
                if let Some(end_offset) = find_ascii_ci(&current[open_end..], &close) {
                    let abs_end = open_end + end_offset + close.len();
                    current.replace_range(start..abs_end, " ");
                } else {
                    current.truncate(start);
                    break;
                }
            }
        }
        // Generic unwrap: remove any remaining <...> tags but keep inner text.
        let mut search_start = 0;
        loop {
            let Some(rel_start) = current[search_start..].find('<') else {
                break;
            };
            let start = search_start + rel_start;
            let Some(end) = current[start..].find('>') else {
                break;
            };
            let abs_end = start + end + 1;
            let after_lt = current[start + 1..].chars().next().unwrap_or(' ');
            if after_lt.is_ascii_alphabetic() || after_lt == '/' || after_lt == '!' {
                current.replace_range(start..abs_end, " ");
                search_start = start;
            } else {
                search_start = abs_end;
            }
            if current.len() > 10000 {
                break;
            }
        }
        return finish_label(&current);
    }
    finish_label(raw)
}

/// Single-allocation finish pass for `sanitize_label`: ANSI strip (borrowed
/// when there is no ESC), whitespace collapse, control-char removal, and
/// the boilerplate suppressions. Equivalent to the old
/// split-whitespace-join plus control-filter pipeline: after collapsing,
/// the only surviving whitespace is ' ', and the suppression patterns are
/// pure ASCII so `eq_ignore_ascii_case` matches `to_lowercase` + compare
/// on them without a whole-message lowercase copy.
fn finish_label(text: &str) -> String {
    let stripped;
    let no_ansi: &str = if text.contains('\x1b') {
        stripped = strip_ansi(text);
        &stripped
    } else {
        text
    };
    let mut collapsed = String::with_capacity(no_ansi.len().min(1024));
    let mut pending_space = false;
    for c in no_ansi.chars() {
        // Whitespace first: newlines/tabs are also control characters, and
        // dropping them here would glue words together ("hello\nworld" must
        // collapse to "hello world", not "helloworld").
        if c.is_whitespace() {
            pending_space = true;
            continue;
        }
        if c.is_control() {
            continue;
        }
        if pending_space && !collapsed.is_empty() {
            collapsed.push(' ');
        }
        pending_space = false;
        collapsed.push(c);
    }
    if collapsed.is_empty() {
        return String::new();
    }
    if starts_ascii_ci(&collapsed, "# agents.md")
        || find_ascii_ci(&collapsed, "global agent preferences").is_some()
        || starts_ascii_ci(&collapsed, "you are a reminder observer")
    {
        return String::new();
    }
    let printable = collapsed.as_str();
    if printable.chars().count() <= MAX_LABEL_CHARS {
        return printable.to_string();
    }
    // Suffix-preserving truncation: keep head and tail to preserve distinguishing suffix.
    const TAIL_LEN: usize = 40;
    let head_len = MAX_LABEL_CHARS.saturating_sub(TAIL_LEN + 1);
    let head_raw: String = printable.chars().take(head_len).collect();
    let head = if let Some(pos) = head_raw.rfind(' ') {
        if pos > 80 {
            head_raw[..pos].to_string()
        } else {
            head_raw
        }
    } else {
        head_raw
    };
    let rev_tail: String = printable.chars().rev().take(TAIL_LEN).collect();
    let tail_raw: String = rev_tail.chars().rev().collect();
    let tail = if let Some(pos) = tail_raw.find(' ') {
        tail_raw[pos + 1..].trim().to_string()
    } else {
        tail_raw.trim().to_string()
    };
    if tail.is_empty() {
        let mut out = head;
        out.push('…');
        return out;
    }
    let mut tail = tail;
    while head.chars().count() + 1 + tail.chars().count() > MAX_LABEL_CHARS {
        if let Some(pos) = tail.find(' ') {
            tail = tail[pos + 1..].trim().to_string();
        } else if tail.chars().count() > 10 {
            tail = tail.chars().skip(tail.chars().count() - 10).collect();
        } else {
            break;
        }
    }
    format!("{}…{}", head, tail)
}

/// ASCII case-insensitive substring search over the original string's bytes.
/// Tag patterns are pure ASCII, so byte-wise matching keeps every index in the
/// original string's coordinates: Unicode `to_lowercase` can shift byte
/// offsets (e.g. U+0130 folds 2 bytes into 3), which would make `replace_range`
/// or `truncate` panic on a non-char-boundary. Every match starts at a `<`
/// byte, which is always a char boundary in UTF-8.
fn find_ascii_ci(haystack: &str, needle: &str) -> Option<usize> {
    let hay = haystack.as_bytes();
    let ndl = needle.as_bytes();
    if ndl.is_empty() || hay.len() < ndl.len() {
        return None;
    }
    (0..=hay.len() - ndl.len()).find(|&i| hay[i..i + ndl.len()].eq_ignore_ascii_case(ndl))
}

/// ASCII case-insensitive prefix test. Uses `get` so a needle length that
/// lands mid-char safely returns false instead of panicking.
fn starts_ascii_ci(haystack: &str, needle: &str) -> bool {
    haystack
        .get(..needle.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(needle))
}

fn strip_ansi(input: &str) -> String {
    if !input.contains('\x1b') {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b {
            // CSI: ESC [ ... letter
            if i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                i += 2;
                while i < bytes.len() && !bytes[i].is_ascii_alphabetic() {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
                continue;
            }
            // OSC: ESC ] ... BEL or ESC \
            if i + 1 < bytes.len() && bytes[i + 1] == b']' {
                i += 2;
                while i < bytes.len()
                    && bytes[i] != 0x07
                    && !(bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'\\')
                {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == 0x07 {
                    i += 1;
                } else if i + 1 < bytes.len() && bytes[i] == 0x1b {
                    i += 2;
                }
                continue;
            }
            i += 1;
            continue;
        }
        // Copy one full UTF-8 scalar value: pushing a single byte as char
        // would corrupt multi-byte sequences (e.g. emoji) into mojibake and
        // inflate char counts used by truncation. `i` always rests on a char
        // boundary here because every skipped escape sequence is pure ASCII.
        let Some(ch) = input[i..].chars().next() else {
            break;
        };
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn opencode_title_for_session(db_path: &str, session_id: &str) -> Option<String> {
    let path = Path::new(db_path);
    if !path.is_file() {
        return None;
    }
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    conn.busy_timeout(Duration::from_secs(1)).ok()?;
    let mut stmt = conn
        .prepare("SELECT title FROM session WHERE id = ?1")
        .ok()?;
    let title: Option<String> = stmt
        .query_row(params![session_id], |row| row.get(0))
        .optional()
        .ok()
        .flatten()
        .filter(|t: &String| !t.trim().is_empty());
    title
}

fn grok_title_for_session(updates_path: &str) -> Option<String> {
    let path = Path::new(updates_path);
    let parent = path.parent()?;
    let summary_path = parent.join("summary.json");
    let contents = std::fs::read_to_string(&summary_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&contents).ok()?;
    for key in [
        "generated_title",
        "generatedTitle",
        "title",
        "session_summary",
        "sessionSummary",
        "summary",
    ] {
        if let Some(title) = value
            .get(key)
            .and_then(|v| v.as_str())
            .filter(|s: &&str| !s.trim().is_empty())
        {
            return Some(title.to_string());
        }
        if let Some(title) = value
            .pointer(&format!("/info/{key}"))
            .and_then(|v| v.as_str())
            .filter(|s: &&str| !s.trim().is_empty())
        {
            return Some(title.to_string());
        }
    }
    value
        .get("info")
        .and_then(|info| info.get("generated_title").or_else(|| info.get("title")))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .or_else(|| {
            value
                .pointer("/info/cwd")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        })
}

fn jcode_label_from_file(path: &str) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&contents).ok()?;
    let messages = value.get("messages")?.as_array()?;
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role != "user" {
            continue;
        }
        let content = msg.get("content")?;
        let mut texts = Vec::new();
        if let Some(arr) = content.as_array() {
            for block in arr {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    texts.push(text.to_string());
                } else if let Some(text) = block.as_str() {
                    texts.push(text.to_string());
                }
            }
        } else if let Some(text) = content.as_str() {
            texts.push(text.to_string());
        }
        let combined = texts.join("\n");
        if combined.trim().is_empty() {
            continue;
        }
        // Skip pure <system-reminder> messages – look for next user message.
        if sanitize_label(&combined).is_empty() {
            continue;
        }
        return Some(combined);
    }
    None
}

/// Parent linkage is the only subagent signal: the `agent` column records
/// the selected agent (build/plan/custom), so a plan-mode session without
/// a parent is still interactive.
fn opencode_session_has_parent(db_path: &str, session_id: &str) -> bool {
    let path = Path::new(db_path);
    if !path.is_file() {
        return false;
    }
    let conn = match Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let _ = conn.busy_timeout(Duration::from_secs(1));
    let check = || -> Option<bool> {
        let mut stmt = conn
            .prepare("SELECT parent_id FROM session WHERE id = ?1")
            .ok()?;
        let parent: Option<String> = stmt
            .query_row(params![session_id], |row| row.get(0))
            .optional()
            .ok()
            .flatten();
        Some(parent.is_some_and(|p| !p.trim().is_empty()))
    };
    check().unwrap_or(false)
}

fn opencode_cwd_for_session(db_path: &str, session_id: &str) -> Option<String> {
    let path = Path::new(db_path);
    if !path.is_file() {
        return None;
    }
    let conn = match Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(c) => c,
        Err(_) => return None,
    };
    let _ = conn.busy_timeout(Duration::from_secs(1));

    let mut stmt = conn
        .prepare("SELECT directory, project_id FROM session WHERE id = ?1")
        .ok()?;
    let row: Option<(Option<String>, Option<String>)> = stmt
        .query_row(params![session_id], |row| Ok((row.get(0)?, row.get(1)?)))
        .optional()
        .ok()
        .flatten();

    let (directory, project_id) = row?;
    if let Some(dir) = directory.filter(|d| !d.trim().is_empty()) {
        return Some(dir);
    }

    if let Some(pid) = project_id.filter(|p| !p.trim().is_empty()) {
        let mut proj_stmt = conn
            .prepare("SELECT worktree FROM project WHERE id = ?1")
            .ok()?;
        let worktree: Option<String> = proj_stmt
            .query_row(params![pid], |row| row.get(0))
            .optional()
            .ok()
            .flatten();
        if let Some(wt) = worktree.filter(|w| !w.trim().is_empty()) {
            return Some(wt);
        }
    }

    None
}

/// Memoized OpenCode source-database lookups for one writer instance. Titles,
/// parent links, and working directories are immutable per session, so caching
/// collapses repeated reads of the same database and rows.
#[derive(Default)]
struct OpencodeLookupCache {
    titles: HashMap<(String, String), String>,
    parented: HashMap<(String, String), bool>,
    cwds: HashMap<(String, String), String>,
}

impl OpencodeLookupCache {
    fn title(&mut self, db_path: &str, session_id: &str) -> Option<String> {
        let key = (db_path.to_string(), session_id.to_string());
        if let Some(cached) = self.titles.get(&key) {
            return Some(cached.clone());
        }
        if let Some(title) = opencode_title_for_session(db_path, session_id) {
            self.titles.insert(key, title.clone());
            Some(title)
        } else {
            None
        }
    }

    fn has_parent(&mut self, db_path: &str, session_id: &str) -> bool {
        let key = (db_path.to_string(), session_id.to_string());
        if let Some(&cached) = self.parented.get(&key) {
            return cached;
        }
        let has = opencode_session_has_parent(db_path, session_id);
        if has {
            self.parented.insert(key, true);
        }
        has
    }

    fn cwd(&mut self, db_path: &str, session_id: &str) -> Option<String> {
        let key = (db_path.to_string(), session_id.to_string());
        if let Some(cached) = self.cwds.get(&key) {
            return Some(cached.clone());
        }
        if let Some(cwd) = opencode_cwd_for_session(db_path, session_id) {
            self.cwds.insert(key, cwd.clone());
            Some(cwd)
        } else {
            None
        }
    }
}

fn extract_session_label(
    source: SourceKind,
    source_path: &str,
    session_id: &str,
    first_user_text: Option<&str>,
    _cwd: Option<&str>,
    opencode: &mut OpencodeLookupCache,
) -> Option<String> {
    let raw = match source {
        SourceKind::Opencode => {
            if let Some(title) = opencode.title(source_path, session_id)
                && !title.trim().is_empty()
            {
                title
            } else {
                first_user_text?.to_string()
            }
        }
        SourceKind::Grok => {
            if let Some(title) = grok_title_for_session(source_path)
                && !title.trim().is_empty()
            {
                title
            } else {
                first_user_text?.to_string()
            }
        }
        SourceKind::Jcode => {
            if let Some(text) = jcode_label_from_file(source_path) {
                text
            } else {
                first_user_text?.to_string()
            }
        }
        _ => first_user_text?.to_string(),
    };
    let label = sanitize_label(&raw);
    if label.is_empty() { None } else { Some(label) }
}

fn infer_session_kind(
    source: SourceKind,
    source_path: &str,
    session_id: &str,
    initial_kind: Option<&str>,
    cwd: Option<&str>,
    first_user_text: Option<&str>,
    opencode: &mut OpencodeLookupCache,
) -> Option<String> {
    if let Some(kind) = initial_kind
        && kind != "main"
        && !kind.is_empty()
    {
        return Some(kind.to_string());
    }
    match source {
        SourceKind::Jcode => {
            // Same worker-sandbox rule as the parser: a bare /tmp cwd is
            // not evidence, only a sandbox leaf name is.
            if let Some(cwd) = cwd
                && jcode_tmp_cwd_is_worker_sandbox(cwd)
            {
                return Some("subagent".to_string());
            }
            if let Some(text) = first_user_text {
                // Shared with the jcode parser directive scan; see
                // `crate::types::jcode_text_is_subagent_directive`.
                if jcode_text_is_subagent_directive(&text.to_lowercase()) {
                    return Some("subagent".to_string());
                }
            }
        }
        SourceKind::Opencode => {
            // Same value the parser stores, so parse-time and backfill
            // classification can never disagree.
            if opencode.has_parent(source_path, session_id) {
                return Some("fork".to_string());
            }
        }
        SourceKind::Grok => {
            // Same rule the grok parser stores (`grok_subagent_kind`), so
            // parse-time and backfill classification can never disagree.
            if let Some(kind) = crate::sources::grok::grok_subagent_kind(
                crate::sources::grok::session_kind(Path::new(source_path)).as_deref(),
            ) {
                return Some(kind);
            }
        }
        // Match whole path components (like the Cursor parser's
        // `is_subagent_transcript`): a bare substring would false-positive
        // on projects such as `my-subagents-tool`.
        SourceKind::Muse | SourceKind::Claude | SourceKind::Cursor => {
            let normalized = source_path.replace('\\', "/");
            if normalized
                .split('/')
                .any(|c| c == "subagents" || c == "subagent")
            {
                return Some("subagent".to_string());
            }
            // Same agent-file convention as the Claude parser's
            // `is_agent_transcript`: applies to any of these sources.
            if source == SourceKind::Claude
                && let Some(name) = normalized.rsplit('/').next()
                && name.starts_with("agent-")
                && name.ends_with(".jsonl")
            {
                return Some("subagent".to_string());
            }
        }
        SourceKind::Codex => {}
        _ => {}
    }
    Some("main".to_string())
}

pub fn analytics_path(state_dir: &Path) -> PathBuf {
    state_dir.join("analytics.sqlite")
}

pub fn rebuild_from_records(
    path: impl AsRef<Path>,
    records: impl IntoIterator<Item = Record>,
) -> Result<()> {
    let mut writer = AnalyticsWriter::open(path)?;
    writer.clear()?;
    for record in records {
        writer.record(&record)?;
    }
    writer.flush()?;
    writer.store.mark_complete()
}

pub fn backfill_from_index(
    path: impl AsRef<Path>,
    index: &crate::index::SearchIndex,
) -> Result<()> {
    let mut writer = AnalyticsWriter::open(path)?;
    writer.clear()?;
    index
        .for_each_record(|record| {
            writer.record(&record)?;
            Ok(())
        })
        .context("read records for analytics backfill")?;
    writer.flush()?;
    writer.store.mark_complete()
}

#[allow(clippy::too_many_arguments)]
fn session_selection_sql(
    source: Option<SourceFilter>,
    project: Option<&str>,
    cwd: Option<&str>,
    since_ms: Option<u64>,
    kind: Option<SessionKindFilter>,
    session_id: Option<&str>,
    source_path: Option<&str>,
) -> (String, Vec<rusqlite::types::Value>) {
    let mut clauses = Vec::new();
    let mut values: Vec<rusqlite::types::Value> = Vec::new();

    if let Some(session_id) = session_id {
        clauses.push("session_id = ?".to_string());
        values.push(rusqlite::types::Value::Text(session_id.to_string()));
    }
    if let Some(source_path) = source_path {
        clauses.push("source_path = ?".to_string());
        values.push(rusqlite::types::Value::Text(source_path.to_string()));
    }

    if let Some(source) = source {
        let labels = source.storage_labels();
        let placeholders = std::iter::repeat_n("?", labels.len())
            .collect::<Vec<_>>()
            .join(", ");
        clauses.push(format!("source IN ({placeholders})"));
        values.extend(
            labels
                .iter()
                .map(|label| rusqlite::types::Value::Text((*label).to_string())),
        );
    }
    if let Some(project) = project {
        clauses.push(format!("{REPOSITORY_PROJECT_SQL} = ?"));
        values.push(rusqlite::types::Value::Text(project.to_string()));
    }
    if let Some(cwd) = cwd {
        let trimmed = cwd.trim_end_matches('/');
        let root = if trimmed.is_empty() {
            "/".to_string()
        } else {
            trimmed.to_string()
        };
        // Escape LIKE wildcards so a path like /tmp/foo_bar doesn't also
        // match sessions under /tmp/fooXbar.
        let escaped = root
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let prefix = if root == "/" {
            "/%".to_string()
        } else {
            format!("{escaped}/%")
        };
        clauses.push(
            "(cwd = ? OR cwd LIKE ? ESCAPE '\\' OR git_root = ? OR git_common_dir = ? OR git_common_dir = ? || '/.git')"
                .to_string(),
        );
        values.push(rusqlite::types::Value::Text(root.clone()));
        values.push(rusqlite::types::Value::Text(prefix));
        values.push(rusqlite::types::Value::Text(root.clone()));
        values.push(rusqlite::types::Value::Text(root.clone()));
        values.push(rusqlite::types::Value::Text(root));
    }
    if let Some(since_ms) = since_ms {
        clauses.push("last_at >= ?".to_string());
        values.push(rusqlite::types::Value::Integer(since_ms as i64));
    }
    if let Some(kind) = kind {
        match kind {
            SessionKindFilter::Primary => {
                clauses
                    .push("(conversation_kind IS NULL OR conversation_kind = 'main')".to_string());
            }
            SessionKindFilter::Subagent => {
                clauses.push(
                        "conversation_kind IS NOT NULL AND conversation_kind NOT IN ('main', 'guardian_review')".to_string(),
                    );
            }
            SessionKindFilter::Regular => {
                clauses.push(
                    "(conversation_kind IS NULL OR conversation_kind != 'guardian_review')"
                        .to_string(),
                );
            }
            SessionKindFilter::All => {}
        }
    }
    let predicate = if clauses.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", clauses.join(" AND "))
    };
    (predicate, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use crate::types::RecordLinks;
    use std::fs;

    #[cfg(unix)]
    #[test]
    fn timed_out_child_is_killed_and_reaped() {
        let _guard = env_lock();
        let child = Command::new("sleep")
            .arg("30")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn child");
        let pid = child.id();

        assert!(child_output_before(child, Instant::now() + Duration::from_millis(20)).is_none());
        assert!(
            !Command::new("kill")
                .args(["-0", &pid.to_string()])
                .stderr(Stdio::null())
                .status()
                .expect("check child")
                .success()
        );
    }

    #[test]
    fn writable_connections_use_durable_wal_commits() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store =
            AnalyticsStore::open(tmp.path().join("analytics.sqlite")).expect("open analytics");

        let journal_mode: String = store
            .conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .expect("journal mode");
        let synchronous: i64 = store
            .conn
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .expect("synchronous mode");

        assert_eq!(journal_mode, "wal");
        assert_eq!(synchronous, 2, "FULL synchronous mode");
    }

    fn record(project: &str, session_id: &str, source_path: &Path, ts: u64) -> Record {
        Record {
            source: SourceKind::Codex,
            doc_id: ts,
            ts,
            project: project.to_string(),
            session_id: session_id.to_string(),
            turn_id: ts as u32,
            role: "user".to_string(),
            text: "hello".to_string(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links: RecordLinks::default(),
            source_path: source_path.to_string_lossy().to_string(),
        }
    }

    #[test]
    fn detailed_sessions_exact_selectors_apply_before_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let store = AnalyticsStore::open(tmp.path().join("analytics.sqlite")).unwrap();
        store.conn.execute_batch(
            "insert into sessions (source, session_id, source_path, project, started_at, last_at) values
             ('codex', 'shared', '/old', 'target', 1, 1),
             ('codex', 'shared', '/new', 'target', 2, 2),
             ('codex', 'other', '/old', 'target', 3, 3),
             ('claude', 'shared', '/old', 'target', 4, 4);",
        ).unwrap();
        let query = |session_id, source_path| {
            store
                .query_sessions_detailed_selected(
                    Some(SourceFilter::Codex),
                    None,
                    None,
                    None,
                    None,
                    session_id,
                    source_path,
                    Some(1),
                )
                .unwrap()
        };
        let exact = query(Some("shared"), Some("/old"));
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].last_at, 1);
        assert_eq!(exact[0].source, SourceKind::Codex);
        assert_eq!(query(Some("shared"), None)[0].source_path, "/new");
        assert_eq!(query(None, Some("/old"))[0].session_id, "other");
        assert!(query(Some("missing"), Some("/old")).is_empty());
        assert!(query(Some("shared"), Some("/old%")).is_empty());
        assert!(
            store
                .query_sessions_detailed_selected(
                    Some(SourceFilter::Codex),
                    Some("different"),
                    None,
                    None,
                    None,
                    Some("shared"),
                    Some("/old"),
                    Some(1),
                )
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn project_summaries_count_the_full_index_and_group_repositories() {
        let tmp = tempfile::tempdir().unwrap();
        let store = AnalyticsStore::open(tmp.path().join("analytics.sqlite")).unwrap();
        store.conn.execute_batch("begin").unwrap();
        for index in 0..250 {
            store.conn.execute(
                "insert into sessions (source, session_id, source_path, project, repo_project, started_at, last_at)
                 values ('codex', ?1, ?2, ?3, 'repo', 0, ?4)",
                params![format!("session-{index}"), format!("/path/{index}"), format!("worktree-{index}"), index + 1],
            ).unwrap();
        }
        store.conn.execute_batch(
            "insert into sessions (source, session_id, source_path, project, repo_project, started_at, last_at) values
             ('claude', 'session-0', '/path/0', 'raw', 'repo', 0, 999),
             ('codex', 'session-0', '/other-path', 'raw', 'repo', 0, 998),
             ('codex', 'older', '/older', 'older', '', 0, 12),
             ('codex', 'blank', '/blank', '   ', null, 0, 5000);
             insert into sessions (source, session_id, source_path, project, repo_project, started_at, last_at, conversation_kind) values
             ('codex', 'review', '/review', 'raw', 'repo', 0, 9000, 'guardian_review'),
             ('codex', 'review-only', '/review-only', 'raw', 'review-only', 0, 9001, 'guardian_review'),
             ('codex', 'unfiled-review', '/unfiled-review', 'raw', null, 0, 9002, 'guardian_review'),
             ('codex', 'agent', '/agent', 'raw', 'repo', 0, 900, 'subagent');
             commit;",
        ).unwrap();
        let summaries = store.query_project_summaries(None).unwrap();
        assert_eq!(
            summaries,
            vec![
                ProjectSummary {
                    project: UNFILED_PROJECT.into(),
                    session_count: 2,
                    last_at: Some(5000)
                },
                ProjectSummary {
                    project: "repo".into(),
                    session_count: 253,
                    last_at: Some(999)
                },
            ]
        );
        for summary in &summaries {
            let matching = store
                .query_sessions_detailed_filtered(
                    None,
                    Some(&summary.project),
                    None,
                    None,
                    Some(SessionKindFilter::Regular),
                    None,
                )
                .unwrap();
            assert_eq!(matching.len() as u64, summary.session_count);
        }
        let codex = store
            .query_project_summaries(Some(SourceFilter::Codex))
            .unwrap();
        assert_eq!(codex[1].session_count, 252);
        assert_eq!(codex[1].last_at, Some(998));
    }

    #[test]
    fn project_summaries_sort_ties_and_preserve_filter_keys_and_unknown_dates() {
        let tmp = tempfile::tempdir().unwrap();
        let store = AnalyticsStore::open(tmp.path().join("analytics.sqlite")).unwrap();
        store.conn.execute_batch(
            "insert into sessions (source, session_id, source_path, project, repo_project, started_at, last_at) values
             ('codex', 'b', '/b', 'raw', 'b', 0, 10),
             ('codex', 'a', '/a', 'raw', 'a', 0, 10),
             ('codex', 'zero', '/zero', 'raw', 'unknown', 0, 0),
             ('codex', 'negative', '/negative', 'raw', 'unknown', 0, -1),
             ('codex', 'encoded', '/encoded', 'raw', '-Users-nico-Code-project', 0, 0);",
        ).unwrap();
        let rows = store.query_project_summaries(None).unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row.project.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "-Users-nico-Code-project", "unknown"]
        );
        assert_eq!(rows[2].last_at, None);
        assert_eq!(rows[3].last_at, None);
        assert_eq!(rows[3].session_count, 2);
    }

    #[test]
    fn display_project_decodes_path_shaped_project_slugs() {
        assert_eq!(display_project_name("-Users-nico-Code"), "~/Code");
        assert_eq!(
            display_project_name("-Users-nico-Code-sidequery-backend"),
            "sidequery-backend"
        );
        assert_eq!(display_project_name("model-serving"), "model-serving");
        let _guard = crate::test_support::EnvVarGuard::set(&[("HOME", Some("/Users/joe"))]);
        assert_eq!(
            display_project_name("joe-Developer-continual-agent-mvp"),
            "continual-agent-mvp"
        );
        assert_eq!(
            display_project_name("-Users-joe-Developer-BenchBox"),
            "BenchBox"
        );
    }

    #[test]
    fn analytics_writer_rolls_records_up_to_sessions() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let transcript = tmp.path().join("session.jsonl");
        fs::write(
            &transcript,
            format!(
                "{{\"type\":\"session_meta\",\"payload\":{{\"cwd\":\"{}\"}}}}\n",
                tmp.path().display()
            ),
        )
        .expect("write transcript");
        let db = tmp.path().join("analytics.sqlite");
        let mut writer = AnalyticsWriter::open(&db).expect("open analytics");
        writer
            .record(&record("memex", "s1", &transcript, 10))
            .expect("record");
        writer
            .record(&record("memex", "s1", &transcript, 20))
            .expect("record");
        writer.flush().expect("flush");

        let store = AnalyticsStore::open(&db).expect("open store");
        let rows = store
            .query_sessions(None, None, None, ProjectGrouping::Flat, None)
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].session_id, "s1");
        assert_eq!(rows[0].message_count, 2);
        assert_eq!(rows[0].last_at, 20);
    }

    #[test]
    fn detailed_sessions_filter_by_cwd_prefix() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo = tmp.path().join("repo");
        let nested = repo.join("crates/core");
        let other = tmp.path().join("other");
        fs::create_dir_all(&nested).expect("mkdir");
        fs::create_dir_all(&other).expect("mkdir");
        let mut transcripts = Vec::new();
        for (name, cwd) in [("in.jsonl", &nested), ("out.jsonl", &other)] {
            let transcript = tmp.path().join(name);
            fs::write(
                &transcript,
                format!(
                    "{{\"type\":\"session_meta\",\"payload\":{{\"cwd\":\"{}\"}}}}\n",
                    cwd.display()
                ),
            )
            .expect("write transcript");
            transcripts.push(transcript);
        }
        let db = tmp.path().join("analytics.sqlite");
        let mut writer = AnalyticsWriter::open(&db).expect("open analytics");
        writer
            .record(&record("repo", "s-in", &transcripts[0], 10))
            .expect("record");
        writer
            .record(&record("other", "s-out", &transcripts[1], 20))
            .expect("record");
        writer.flush().expect("flush");

        let store = AnalyticsStore::open_read_only(&db).expect("open read only");
        let all = store
            .query_sessions_detailed(None, None, None, None, None)
            .expect("all sessions");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].session_id, "s-out");

        let scoped = store
            .query_sessions_detailed(
                None,
                None,
                Some(repo.to_string_lossy().as_ref()),
                None,
                None,
            )
            .expect("scoped sessions");
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].session_id, "s-in");
        assert_eq!(scoped[0].cwd.as_deref(), Some(&*nested.to_string_lossy()));
    }

    #[test]
    fn detailed_sessions_cwd_filter_escapes_like_wildcards() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let target = tmp.path().join("foo_bar");
        let sibling = tmp.path().join("fooXbar");
        fs::create_dir_all(target.join("sub")).expect("mkdir");
        fs::create_dir_all(sibling.join("sub")).expect("mkdir");
        let mut transcripts = Vec::new();
        for (name, cwd) in [
            ("target.jsonl", target.join("sub")),
            ("sibling.jsonl", sibling.join("sub")),
        ] {
            let transcript = tmp.path().join(name);
            fs::write(
                &transcript,
                format!(
                    "{{\"type\":\"session_meta\",\"payload\":{{\"cwd\":\"{}\"}}}}\n",
                    cwd.display()
                ),
            )
            .expect("write transcript");
            transcripts.push(transcript);
        }
        let db = tmp.path().join("analytics.sqlite");
        let mut writer = AnalyticsWriter::open(&db).expect("open analytics");
        writer
            .record(&record("foo_bar", "s-target", &transcripts[0], 10))
            .expect("record");
        writer
            .record(&record("fooXbar", "s-sibling", &transcripts[1], 20))
            .expect("record");
        writer.flush().expect("flush");

        let store = AnalyticsStore::open_read_only(&db).expect("open read only");
        let scoped = store
            .query_sessions_detailed(
                None,
                None,
                Some(target.to_string_lossy().as_ref()),
                None,
                None,
            )
            .expect("scoped sessions");
        assert_eq!(
            store
                .count_sessions_selected(
                    None,
                    None,
                    Some(target.to_string_lossy().as_ref()),
                    None,
                    None,
                    None,
                    None
                )
                .unwrap(),
            1
        );
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].session_id, "s-target");
    }

    #[test]
    fn read_only_store_rejects_writes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db = tmp.path().join("analytics.sqlite");
        drop(AnalyticsStore::open(&db).expect("initialize analytics"));

        let store = AnalyticsStore::open_read_only(&db).expect("open read only");

        assert!(store.mark_complete().is_err());
    }

    #[test]
    fn project_queries_are_distinct_and_timeline_projection_is_narrow() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let source_a = tmp.path().join("a.jsonl");
        let source_b = tmp.path().join("b.jsonl");
        fs::write(&source_a, "").expect("source a");
        fs::write(&source_b, "").expect("source b");
        let db = tmp.path().join("analytics.sqlite");
        rebuild_from_records(
            &db,
            [
                record("alpha", "s1", &source_a, 10),
                record("alpha", "s2", &source_b, 20),
            ],
        )
        .expect("rebuild");
        let store = AnalyticsStore::open_read_only(&db).expect("open read only");

        assert_eq!(
            store
                .query_projects(None, ProjectGrouping::Flat)
                .expect("projects"),
            vec!["alpha"]
        );
        assert_eq!(
            store
                .query_project_timestamps(None, Some(15), ProjectGrouping::Flat)
                .expect("timestamps"),
            vec![("alpha".to_string(), 20)]
        );
    }

    #[test]
    fn source_timestamps_apply_activity_filters_without_a_result_limit() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db = tmp.path().join("analytics.sqlite");
        let records = [("alpha", "s1", 10), ("alpha", "s2", 20), ("beta", "s3", 30)]
            .into_iter()
            .map(|(project, session, ts)| {
                record(
                    project,
                    session,
                    &tmp.path().join(format!("{session}.jsonl")),
                    ts,
                )
            });
        rebuild_from_records(&db, records).expect("rebuild");
        let store = AnalyticsStore::open_read_only(&db).expect("open read only");

        assert_eq!(
            store
                .query_source_timestamps_filtered(
                    Some(SourceFilter::Codex),
                    Some(15),
                    Some(25),
                    Some("alpha"),
                    ProjectGrouping::Flat,
                    None,
                )
                .expect("filtered activity"),
            vec![(SourceKind::Codex, 20)]
        );
    }

    #[test]
    fn repository_project_filter_uses_expression_index() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db = tmp.path().join("analytics.sqlite");
        let store = AnalyticsStore::open(&db).expect("open analytics");
        let plan: String = store
            .conn
            .query_row(
                "EXPLAIN QUERY PLAN SELECT source FROM sessions
                 WHERE COALESCE(NULLIF(repo_project, ''), 'Unfiled') = ?1
                 ORDER BY last_at DESC LIMIT 200",
                params!["memex"],
                |row| row.get(3),
            )
            .expect("query plan");

        assert!(
            plan.contains("sessions_repository_project_last_at_idx"),
            "{plan}"
        );
    }

    #[test]
    fn repository_grouping_buckets_non_git_sessions_as_unfiled() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let standalone = tmp.path().join("generated-task-name");
        fs::create_dir(&standalone).expect("standalone dir");
        let standalone_transcript = tmp.path().join("standalone.jsonl");
        fs::write(
            &standalone_transcript,
            format!(
                "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"standalone\",\"cwd\":\"{}\"}}}}\n",
                standalone.display()
            ),
        )
        .expect("write standalone transcript");
        let repo_transcript = tmp.path().join("repo.jsonl");
        fs::write(&repo_transcript, "").expect("write repo transcript");
        let db = tmp.path().join("analytics.sqlite");
        rebuild_from_records(
            &db,
            [
                record(
                    "generated-task-name",
                    "standalone",
                    &standalone_transcript,
                    10,
                ),
                record("raw-repo-slug", "repo", &repo_transcript, 20),
            ],
        )
        .expect("rebuild");

        let store = AnalyticsStore::open(&db).expect("open store");
        store
            .conn
            .execute(
                "UPDATE sessions SET repo_project = 'alpha' WHERE session_id = 'repo'",
                [],
            )
            .expect("seed repository project");

        let rows = store
            .query_sessions(None, None, None, ProjectGrouping::Repository, None)
            .expect("repository sessions");
        assert_eq!(rows[0].display_project, "alpha");
        assert_eq!(rows[1].project, "generated-task-name");
        assert_eq!(rows[1].display_project, UNFILED_PROJECT);
        assert_eq!(
            store
                .query_projects(None, ProjectGrouping::Repository)
                .expect("repository projects"),
            vec![UNFILED_PROJECT, "alpha"]
        );

        let unfiled = store
            .query_sessions(
                None,
                None,
                Some(UNFILED_PROJECT),
                ProjectGrouping::Repository,
                None,
            )
            .expect("unfiled sessions");
        assert_eq!(unfiled.len(), 1);
        assert_eq!(unfiled[0].session_id, "standalone");
        assert_eq!(
            store
                .query_sessions_detailed(None, Some(UNFILED_PROJECT), None, None, None)
                .expect("detailed unfiled sessions")
                .len(),
            1
        );
        assert_eq!(
            store
                .query_project_timestamps(None, None, ProjectGrouping::Repository)
                .expect("repository timestamps"),
            vec![(UNFILED_PROJECT.to_string(), 10), ("alpha".to_string(), 20)]
        );

        let session_projects = store
            .query_session_projects(
                &[(
                    SourceKind::Codex,
                    "standalone".to_string(),
                    standalone_transcript.to_string_lossy().to_string(),
                )],
                ProjectGrouping::Repository,
            )
            .expect("session projects");
        assert_eq!(
            session_projects.values().next().map(String::as_str),
            Some(UNFILED_PROJECT)
        );

        let flat = store
            .query_sessions(None, None, None, ProjectGrouping::Flat, None)
            .expect("flat sessions");
        assert_eq!(flat[1].display_project, "generated-task-name");
    }

    #[test]
    fn analytics_schema_version_change_marks_incomplete() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db = tmp.path().join("analytics.sqlite");
        {
            let conn = Connection::open(&db).expect("open sqlite");
            conn.execute_batch(
                r#"
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO meta(key, value) VALUES('schema_version', '1');
                INSERT INTO meta(key, value) VALUES('analytics_complete', '1');
                "#,
            )
            .expect("seed meta");
        }

        let store = AnalyticsStore::open(&db).expect("open store");

        assert!(!store.complete().expect("complete"));
    }

    #[test]
    fn repository_grouping_uses_git_common_dir_project() {
        let _guard = env_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo = tmp.path().join("memex");
        fs::create_dir_all(&repo).expect("repo dir");
        assert!(
            Command::new("git")
                .args(["init"])
                .current_dir(&repo)
                .output()
                .expect("git init")
                .status
                .success()
        );
        let transcript = tmp.path().join("session.jsonl");
        fs::write(
            &transcript,
            format!(
                "{{\"cwd\":\"{}\",\"type\":\"session_meta\",\"payload\":{{\"cwd\":\"{}\"}}}}\n",
                repo.display(),
                repo.display()
            ),
        )
        .expect("write transcript");

        let db = tmp.path().join("analytics.sqlite");
        rebuild_from_records(
            &db,
            [record(
                "memex-claude-worktrees-feature",
                "s1",
                &transcript,
                10,
            )],
        )
        .expect("rebuild");

        let store = AnalyticsStore::open(&db).expect("open store");
        let rows = store
            .query_sessions(None, None, None, ProjectGrouping::Repository, None)
            .expect("query");
        assert_eq!(rows[0].project, "memex-claude-worktrees-feature");
        assert_eq!(rows[0].display_project, "memex");
    }

    #[test]
    fn claude_worktree_path_falls_back_to_parent_repo() {
        assert_eq!(
            claude_worktree_repo_project(
                "/Users/nico/Code/atm-backend/.claude/worktrees/exciting-morse-e2914f"
            )
            .as_deref(),
            Some("atm-backend")
        );
        assert_eq!(
            claude_worktree_repo_project("/Users/nico/Code/atm-backend"),
            None
        );
    }

    #[test]
    fn codex_worktree_path_falls_back_to_repo_directory() {
        let cwd = "/missing/home/.codex/worktrees/8952/memex/crates/core";
        assert_eq!(codex_worktree_repo_project(cwd).as_deref(), Some("memex"));
        let metadata = git_metadata_for_cwd(cwd);
        assert_eq!(metadata.repo_project.as_deref(), Some("memex"));
        assert_eq!(metadata.status, "path-fallback");
        assert_eq!(
            codex_worktree_repo_project("/missing/home/.codex/worktrees/8952"),
            None
        );
        assert_eq!(
            codex_worktree_repo_project("/missing/home/Documents/Codex/2026-09-07/hel"),
            None
        );
    }

    #[test]
    fn repository_grouping_uses_claude_worktree_path_without_local_git() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let transcript = tmp.path().join("session.jsonl");
        fs::write(
            &transcript,
            "{\"cwd\":\"/Users/nico/Code/atm-backend/.claude/worktrees/exciting-morse-e2914f\"}\n",
        )
        .expect("write transcript");

        let db = tmp.path().join("analytics.sqlite");
        rebuild_from_records(
            &db,
            [record(
                "ssh-d4309b74-100f-407e-b64d-31c7160044cd",
                "s1",
                &transcript,
                10,
            )],
        )
        .expect("rebuild");

        let store = AnalyticsStore::open(&db).expect("open store");
        let rows = store
            .query_sessions(None, None, None, ProjectGrouping::Repository, None)
            .expect("query");
        assert_eq!(rows[0].project, "ssh-d4309b74-100f-407e-b64d-31c7160044cd");
        assert_eq!(rows[0].display_project, "atm-backend");
    }

    #[test]
    fn raw_projects_for_repository_finds_distinct_projects() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo_dir = tmp.path().join("memex");
        fs::create_dir_all(repo_dir.join(".git")).expect("mkdir repo .git");
        let wt_path = tmp.path().join("memex.wt-feat");
        let transcript1 = tmp.path().join("session1.jsonl");
        fs::write(
            &transcript1,
            format!("{{\"cwd\":\"{}\"}}\n", wt_path.display()),
        )
        .expect("write transcript1");
        let transcript2 = tmp.path().join("session2.jsonl");
        fs::write(
            &transcript2,
            format!("{{\"cwd\":\"{}\"}}\n", repo_dir.display()),
        )
        .expect("write transcript2");

        let db = tmp.path().join("analytics.sqlite");
        rebuild_from_records(
            &db,
            [
                record("memex.wt-feat", "s1", &transcript1, 10),
                record("memex", "s2", &transcript2, 20),
            ],
        )
        .expect("rebuild");

        let store = AnalyticsStore::open(&db).expect("open store");
        let mut raw = store
            .raw_projects_for_repository("memex")
            .expect("raw projects");
        raw.sort();
        assert_eq!(raw, vec!["memex", "memex.wt-feat"]);
    }

    #[test]
    fn sanitize_label_collapses_whitespace_and_truncates() {
        let raw = "  Hello\n   world   \x1b[31mred\x1b[0m  <system-reminder>ignore</system-reminder>  this is a very long prompt that should be truncated at word boundary because it exceeds the one hundred fifty character limit significantly and we want to ensure ellipsis handling works correctly for display";
        let label = sanitize_label(raw);
        assert!(!label.contains('\n'));
        assert!(!label.contains("\x1b"));
        assert!(!label.contains("ignore"));
        assert!(label.chars().count() <= MAX_LABEL_CHARS);
        assert!(label.ends_with('…') || label.chars().count() < MAX_LABEL_CHARS);
        assert!(label.starts_with("Hello world red"));
    }

    #[test]
    fn analytics_stores_label_from_first_user_message() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let transcript = tmp.path().join("session.jsonl");
        fs::write(
            &transcript,
            format!(
                "{{\"type\":\"session_meta\",\"payload\":{{\"cwd\":\"{}\"}}}}\n",
                tmp.path().display()
            ),
        )
        .expect("write");
        let db = tmp.path().join("analytics.sqlite");
        let mut writer = AnalyticsWriter::open(&db).expect("open");
        let mut rec = record("proj", "s-label", &transcript, 10);
        rec.role = "user".to_string();
        rec.text = "Fix the login bug on the dashboard".to_string();
        rec.links.conversation_kind = Some("main".to_string());
        writer.record(&rec).expect("record");
        writer.flush().expect("flush");
        let store = AnalyticsStore::open_read_only(&db).expect("open ro");
        let rows = store
            .query_sessions_detailed(None, None, None, None, None)
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].label.as_deref(),
            Some("Fix the login bug on the dashboard")
        );
        assert_eq!(rows[0].conversation_kind.as_deref(), Some("main"));
    }

    #[test]
    fn incremental_delta_keeps_original_label_and_kind() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let transcript = tmp.path().join("session.jsonl");
        fs::write(&transcript, "").expect("write");
        let db = tmp.path().join("analytics.sqlite");
        let mut first = record("proj", "s-delta", &transcript, 10);
        first.role = "user".to_string();
        first.text = "REAL FIRST PROMPT about the login bug".to_string();
        first.links.conversation_kind = Some("main".to_string());
        let mut writer = AnalyticsWriter::open(&db).expect("open");
        writer.record(&first).expect("record");
        writer.flush().expect("flush");
        drop(writer);
        // A later incremental run sees only a mid-session message, possibly
        // with a different per-record kind. Neither may clobber the stored row.
        let mut delta = record("proj", "s-delta", &transcript, 20);
        delta.role = "user".to_string();
        delta.text = "now also update the changelog".to_string();
        delta.links.conversation_kind = Some("subagent".to_string());
        let mut writer = AnalyticsWriter::open(&db).expect("reopen");
        writer.record(&delta).expect("record");
        writer.flush().expect("flush");
        let store = AnalyticsStore::open_read_only(&db).expect("open ro");
        let rows = store
            .query_sessions_detailed(None, None, None, None, None)
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].label.as_deref(),
            Some("REAL FIRST PROMPT about the login bug")
        );
        assert_eq!(rows[0].conversation_kind.as_deref(), Some("main"));
    }

    #[test]
    fn main_record_wins_over_compaction_entries() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let transcript = tmp.path().join("session.jsonl");
        fs::write(&transcript, "").expect("write");
        let db = tmp.path().join("analytics.sqlite");
        let mut writer = AnalyticsWriter::open(&db).expect("open");
        // Compaction summary arrives before any main message, as in a
        // resumed Pi transcript: the session is still interactive.
        let mut summary = record("proj", "s-compact", &transcript, 5);
        summary.role = "user".to_string();
        summary.text = "summary of prior work".to_string();
        summary.links.conversation_kind = Some("compaction".to_string());
        writer.record(&summary).expect("record");
        let mut prompt = record("proj", "s-compact", &transcript, 10);
        prompt.role = "user".to_string();
        prompt.text = "please fix the parser".to_string();
        prompt.links.conversation_kind = Some("main".to_string());
        writer.record(&prompt).expect("record");
        writer.flush().expect("flush");
        let store = AnalyticsStore::open_read_only(&db).expect("open ro");
        let rows = store
            .query_sessions_detailed(None, None, None, None, None)
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].conversation_kind.as_deref(), Some("main"));
    }

    #[test]
    fn every_stored_kind_has_a_defined_filter_bucket() {
        // Exercise every stored kind through SQL and the in-memory predicate.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db = tmp.path().join("analytics.sqlite");
        let mut writer = AnalyticsWriter::open(&db).expect("open");
        for (id, kind, ts) in [
            ("s-main", "main", 10),
            ("s-sub", "subagent", 20),
            ("s-fork", "fork", 30),
            ("s-side", "sidechain", 40),
            ("s-compact", "compaction", 50),
            ("s-branch", "branch", 60),
            ("s-review", "guardian_review", 70),
        ] {
            let path = tmp.path().join(format!("{id}.jsonl"));
            fs::write(&path, "").expect("write");
            let mut rec = record("proj", id, &path, ts);
            rec.role = "user".to_string();
            rec.text = format!("task {id}");
            rec.links.conversation_kind = Some(kind.to_string());
            writer.record(&rec).expect("record");
        }
        writer.flush().expect("flush");
        let store = AnalyticsStore::open_read_only(&db).expect("open ro");
        let filtered = |kind| {
            store
                .query_sessions_detailed_filtered(None, None, None, None, Some(kind), None)
                .expect("query")
                .into_iter()
                .map(|row| row.session_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(filtered(SessionKindFilter::Primary), vec!["s-main"]);
        let mut sub = filtered(SessionKindFilter::Subagent);
        sub.sort();
        assert_eq!(
            sub,
            vec!["s-branch", "s-compact", "s-fork", "s-side", "s-sub"]
        );
        assert_eq!(filtered(SessionKindFilter::Regular).len(), 6);
        assert_eq!(filtered(SessionKindFilter::All).len(), 7);
        for filter in [
            SessionKindFilter::Primary,
            SessionKindFilter::Subagent,
            SessionKindFilter::Regular,
            SessionKindFilter::All,
        ] {
            let rows = store
                .query_sessions_filtered(
                    None,
                    None,
                    None,
                    ProjectGrouping::Flat,
                    Some(filter),
                    None,
                )
                .unwrap();
            assert_eq!(rows.len(), filtered(filter).len());
            for row in rows {
                assert!(filter.matches_kind(row.conversation_kind.as_deref()));
            }
            assert_eq!(
                filter.matches_kind(Some("guardian_review")),
                filter == SessionKindFilter::All
            );
        }
        assert_eq!(
            store
                .query_sessions_detailed(None, None, None, None, None)
                .expect("all")
                .len(),
            7
        );
    }

    #[test]
    fn session_cwd_resolves_from_jcode_working_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("session_cwd.json");
        fs::write(
            &path,
            r#"{"id":"s-cwd","working_dir":"/repo/example","messages":[]}"#,
        )
        .expect("write");
        let cwd = resolve_session_cwd_from_parts(
            SourceKind::Jcode,
            &path.to_string_lossy(),
            "s-cwd",
            &mut OpencodeLookupCache::default(),
        );
        assert_eq!(cwd.as_deref(), Some("/repo/example"));

        let standalone =
            resolve_session_cwd_standalone(SourceKind::Jcode, &path.to_string_lossy(), "s-cwd");
        assert_eq!(standalone.as_deref(), Some("/repo/example"));
    }

    #[test]
    fn analytics_query_session_cwd_and_cwds() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db = tmp.path().join("analytics.sqlite");
        let a_path = tmp.path().join("a.jsonl");
        let b_path = tmp.path().join("b.jsonl");
        fs::write(&a_path, "").expect("write");
        fs::write(&b_path, "").expect("write");

        let mut writer = AnalyticsWriter::open(&db).expect("open");
        let mut rec = record("proj", "s-1", &a_path, 10);
        rec.source = SourceKind::Muse;
        rec.role = "user".to_string();
        rec.text = "test query cwd".to_string();
        writer.record(&rec).expect("record");

        // s-2 has NULL cwd and git_root
        let mut rec2 = record("proj", "s-2", &b_path, 20);
        rec2.source = SourceKind::Muse;
        rec2.role = "user".to_string();
        rec2.text = "test null cwd".to_string();
        writer.record(&rec2).expect("record");
        writer.flush().expect("flush");

        // Manually update cwd for s-1
        let conn = rusqlite::Connection::open(&db).expect("open");
        conn.execute(
            "UPDATE sessions SET cwd = '/repo/custom-workspace' WHERE session_id = 's-1'",
            [],
        )
        .expect("update");

        let store = AnalyticsStore::open_read_only(&db).expect("open store");
        let cwd = store
            .query_session_cwd(SourceKind::Muse, "s-1", Some(&a_path.to_string_lossy()))
            .expect("query_session_cwd");
        assert_eq!(cwd.as_deref(), Some("/repo/custom-workspace"));

        // NULL cwd and NULL git_root must return Ok(None) without rusqlite type conversion error
        let null_cwd = store
            .query_session_cwd(SourceKind::Muse, "s-2", Some(&b_path.to_string_lossy()))
            .expect("query_session_cwd null");
        assert_eq!(null_cwd, None);

        // Test chunking with >100 entries
        let mut batch = Vec::new();
        for i in 0..150 {
            batch.push((
                SourceKind::Muse,
                format!("s-batch-{i}"),
                format!("/fake/path/{i}.jsonl"),
            ));
        }
        batch.push((
            SourceKind::Muse,
            "s-1".to_string(),
            a_path.to_string_lossy().to_string(),
        ));
        let cwds = store
            .query_session_cwds(&batch)
            .expect("query_session_cwds");
        assert_eq!(
            cwds.get(&(
                SourceKind::Muse,
                "s-1".to_string(),
                a_path.to_string_lossy().to_string()
            ))
            .map(String::as_str),
            Some("/repo/custom-workspace")
        );

        // Test ambiguous session ID without source_path:
        // If s-1 has a second row with a different cwd, query without source_path must return None
        let c_path = tmp.path().join("c.jsonl");
        fs::write(&c_path, "").expect("write");
        conn.execute(
            "INSERT INTO sessions (source, session_id, source_path, project, started_at, last_at, message_count, cwd)
             VALUES ('muse', 's-1', ?1, 'proj', 10, 10, 1, '/different/worktree')",
            [c_path.to_string_lossy().to_string()],
        )
        .expect("insert");

        let ambiguous = store
            .query_session_cwd(SourceKind::Muse, "s-1", None)
            .expect("query ambiguous");
        assert_eq!(ambiguous, None);
    }

    #[test]
    fn analytics_filters_by_conversation_kind() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let a_path = tmp.path().join("a.jsonl");
        let b_path = tmp.path().join("b.jsonl");
        for p in [&a_path, &b_path] {
            fs::write(p, "").expect("write");
        }
        let db = tmp.path().join("analytics.sqlite");
        let mut writer = AnalyticsWriter::open(&db).expect("open");
        let mut primary = record("proj", "s-primary", &a_path, 10);
        primary.role = "user".to_string();
        primary.text = "primary task".to_string();
        primary.links.conversation_kind = Some("main".to_string());
        let mut sub = record("proj", "s-sub", &b_path, 20);
        sub.role = "user".to_string();
        sub.text = "subagent task".to_string();
        sub.links.conversation_kind = Some("subagent".to_string());
        writer.record(&primary).expect("record");
        writer.record(&sub).expect("record");
        writer.flush().expect("flush");
        let store = AnalyticsStore::open_read_only(&db).expect("open ro");
        let primary_rows = store
            .query_sessions_detailed_filtered(
                None,
                None,
                None,
                None,
                Some(SessionKindFilter::Primary),
                None,
            )
            .expect("primary");
        assert_eq!(primary_rows.len(), 1);
        assert_eq!(primary_rows[0].session_id, "s-primary");
        let sub_rows = store
            .query_sessions_detailed_filtered(
                None,
                None,
                None,
                None,
                Some(SessionKindFilter::Subagent),
                None,
            )
            .expect("sub");
        assert_eq!(sub_rows.len(), 1);
        assert_eq!(sub_rows[0].session_id, "s-sub");
        let all_rows = store
            .query_sessions_detailed(None, None, None, None, None)
            .expect("all");
        assert_eq!(all_rows.len(), 2);

        let primary_ts = store
            .query_source_timestamps_filtered(
                None,
                None,
                None,
                None,
                ProjectGrouping::Flat,
                Some(SessionKindFilter::Primary),
            )
            .expect("primary ts");
        assert_eq!(primary_ts, vec![(SourceKind::Codex, 10)]);

        let sub_ts = store
            .query_source_timestamps_filtered(
                None,
                None,
                None,
                None,
                ProjectGrouping::Flat,
                Some(SessionKindFilter::Subagent),
            )
            .expect("sub ts");
        assert_eq!(sub_ts, vec![(SourceKind::Codex, 20)]);

        let all_ts = store
            .query_source_timestamps_filtered(
                None,
                None,
                None,
                None,
                ProjectGrouping::Flat,
                Some(SessionKindFilter::All),
            )
            .expect("all ts");
        assert_eq!(all_ts.len(), 2);
    }

    #[test]
    fn jcode_tmp_cwd_needs_worker_sandbox_leaf_for_subagent() {
        let _tmp = tempfile::tempdir().expect("tempdir");
        let _transcript = _tmp.path().join("session_session_tmp.json");
        // Need a file that contains cwd; but we also need to test inference via cwd.
        // We'll directly test infer_session_kind helper.
        let infer = |cwd: &str| {
            infer_session_kind(
                SourceKind::Jcode,
                "/tmp/.jcode/sessions/session_tmp.json",
                "session_tmp",
                None,
                Some(cwd),
                Some("hello"),
                &mut OpencodeLookupCache::default(),
            )
        };
        // A bare /tmp cwd is not evidence: users legitimately work in /tmp.
        assert_eq!(infer("/tmp/work").as_deref(), Some("main"));
        assert_eq!(infer("/tmp").as_deref(), Some("main"));
        // A worker-sandbox leaf under /tmp corroborates a spawned worker.
        assert_eq!(
            infer("/private/tmp/bossmode-hygiene-v2-worker-docs").as_deref(),
            Some("subagent")
        );
        let kind2 = infer_session_kind(
            SourceKind::Jcode,
            "/tmp/.jcode/sessions/session_main.json",
            "session_main",
            Some("main"),
            Some("/repo/example"),
            Some("hello"),
            &mut OpencodeLookupCache::default(),
        );
        assert_eq!(kind2.as_deref(), Some("main"));
    }

    #[test]
    fn strip_ansi_preserves_multibyte_unicode() {
        let label = sanitize_label("Fix the 🚀 deploy \x1b[31mred\x1b[0m pipeline ✅ now");
        assert_eq!(label, "Fix the 🚀 deploy red pipeline ✅ now");
        assert!(label.contains('🚀'));
    }

    #[test]
    fn sanitize_label_truncates_unicode_by_chars_not_bytes() {
        let raw = "🚀".repeat(200);
        let label = sanitize_label(&raw);
        assert!(label.chars().count() <= MAX_LABEL_CHARS);
        assert!(label.contains('…'));
        assert!(label.starts_with("🚀"));
    }

    #[test]
    fn sanitize_label_with_unicode_case_fold_before_tag_does_not_panic() {
        // U+0130 folds to 2 chars on lowercase; byte offsets computed on the
        // folded copy would be wrong for the original. Must strip safely.
        let raw = "İstanbul \u{130} <SYSTEM-REMINDER>hidden</SYSTEM-REMINDER> visible task";
        let label = sanitize_label(raw);
        assert!(!label.contains("hidden"));
        assert!(!label.contains("SYSTEM-REMINDER"));
        assert!(label.contains("visible task"));
    }

    #[test]
    fn sanitize_label_preserves_lone_comparison_brackets() {
        let raw = "a < b and c > d comparison";
        let label = sanitize_label(raw);
        assert_eq!(label, "a < b and c > d comparison");
    }

    #[test]
    fn session_conversation_kind_resolves_stored_row() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let transcript = tmp.path().join("session.jsonl");
        fs::write(&transcript, "").expect("write");
        let db = tmp.path().join("analytics.sqlite");
        let mut writer = AnalyticsWriter::open(&db).expect("open");
        let mut rec = record("proj", "s-kind", &transcript, 10);
        rec.role = "user".to_string();
        rec.text = "do the thing".to_string();
        rec.links.conversation_kind = Some("fork".to_string());
        writer.record(&rec).expect("record");
        writer.flush().expect("flush");
        let store = AnalyticsStore::open_read_only(&db).expect("open ro");
        assert_eq!(
            store.session_conversation_kind(
                SourceKind::Codex.storage_label(),
                "s-kind",
                &transcript.to_string_lossy()
            ),
            Some("fork".to_string())
        );
        assert_eq!(
            store.session_conversation_kind(
                SourceKind::Codex.storage_label(),
                "s-missing",
                &transcript.to_string_lossy()
            ),
            None
        );
    }

    #[test]
    fn sanitize_label_keeps_word_boundary_on_control_whitespace() {
        assert_eq!(sanitize_label("hello\nworld"), "hello world");
        assert_eq!(sanitize_label("hello\tworld"), "hello world");
        assert_eq!(sanitize_label("hello\r\nworld"), "hello world");
    }

    #[test]
    fn infer_session_kind_matches_cursor_path_components_only() {
        let mut cache = OpencodeLookupCache::default();
        // Bare substring must not classify: project merely mentions subagents.
        let not_sub = infer_session_kind(
            SourceKind::Cursor,
            "/data/my-subagents-tool/session.json",
            "session",
            Some("main"),
            None,
            None,
            &mut cache,
        );
        assert_eq!(not_sub.as_deref(), Some("main"));
        let is_sub = infer_session_kind(
            SourceKind::Cursor,
            "/data/Cursor/projects/subagents/agent-1/transcript.json",
            "agent-1",
            Some("main"),
            None,
            None,
            &mut cache,
        );
        assert_eq!(is_sub.as_deref(), Some("subagent"));
    }

    #[test]
    fn infer_session_kind_matches_muse_plural_subagents_dir() {
        let mut cache = OpencodeLookupCache::default();
        let kind = infer_session_kind(
            SourceKind::Muse,
            "/data/muse/projects/p/subagents/abc123/stream.jsonl",
            "abc123",
            Some("main"),
            None,
            None,
            &mut cache,
        );
        assert_eq!(kind.as_deref(), Some("subagent"));
    }

    #[test]
    fn infer_session_kind_reads_grok_summary_subagent_family() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let session = tmp.path().join("session-id");
        std::fs::create_dir_all(&session).expect("mkdir");
        let updates = session.join("updates.jsonl");
        std::fs::write(&updates, "").expect("write updates");
        let infer = |summary_json: &str| {
            std::fs::write(session.join("summary.json"), summary_json).expect("write summary");
            infer_session_kind(
                SourceKind::Grok,
                updates.to_str().expect("utf8 path"),
                "session-id",
                None,
                None,
                None,
                &mut OpencodeLookupCache::default(),
            )
        };
        assert_eq!(
            infer(r#"{"session_kind":"subagent","info":{"id":"session-id"}}"#).as_deref(),
            Some("subagent")
        );
        assert_eq!(
            infer(r#"{"session_kind":"subagent_resume","info":{"id":"session-id"}}"#).as_deref(),
            Some("subagent_resume")
        );
        // Absent and non-subagent markers stay interactive.
        assert_eq!(
            infer(r#"{"info":{"id":"session-id"}}"#).as_deref(),
            Some("main")
        );
        assert_eq!(
            infer(r#"{"session_kind":"headless","info":{"id":"session-id"}}"#).as_deref(),
            Some("main")
        );
    }

    #[test]
    fn extract_session_label_falls_back_without_opencode_db() {
        let mut cache = OpencodeLookupCache::default();
        let label = extract_session_label(
            SourceKind::Opencode,
            "/nonexistent/opencode.db",
            "missing",
            Some("  hello world  "),
            None,
            &mut cache,
        );
        assert_eq!(label.as_deref(), Some("hello world"));
    }

    #[test]
    fn worktree_repo_info_resolves_from_git_file_with_commondir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let main_repo = tmp.path().join("main-project");
        let main_git = main_repo.join(".git");
        let wt_admin = main_git.join("worktrees").join("feature-wt");
        fs::create_dir_all(&wt_admin).expect("mkdir wt_admin");
        fs::write(wt_admin.join("commondir"), "../..\n").expect("write commondir");

        let wt_workdir = tmp.path().join("feature-worktree");
        fs::create_dir_all(&wt_workdir).expect("mkdir wt_workdir");
        fs::write(
            wt_workdir.join(".git"),
            format!("gitdir: {}\n", wt_admin.display()),
        )
        .expect("write .git file");

        let info = worktree_repo_project(wt_workdir.to_str().expect("utf8"));
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.repo_project, "main-project");
        assert_eq!(
            info.git_root.as_deref(),
            main_repo
                .canonicalize()
                .ok()
                .map(|p| p.to_string_lossy().to_string())
                .as_deref()
        );
    }

    #[test]
    fn worktree_repo_info_rejects_submodule_without_commondir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let sub_admin = tmp
            .path()
            .join("main-project")
            .join(".git")
            .join("modules")
            .join("submod");
        fs::create_dir_all(&sub_admin).expect("mkdir sub_admin");
        // Note: NO commondir file written to sub_admin!

        let sub_workdir = tmp.path().join("main-project").join("submod");
        fs::create_dir_all(&sub_workdir).expect("mkdir sub_workdir");
        fs::write(
            sub_workdir.join(".git"),
            format!("gitdir: {}\n", sub_admin.display()),
        )
        .expect("write .git file");

        let info = worktree_repo_project(sub_workdir.to_str().expect("utf8"));
        assert_eq!(info, None);
    }

    #[test]
    fn worktree_repo_info_resolves_sibling_worktree_fallback() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo_dir = tmp.path().join("my-backend");
        fs::create_dir_all(repo_dir.join(".git")).expect("mkdir repo .git");

        let wt_path = tmp.path().join("my-backend.wt-hotfix");
        // wt_path does not even have to exist on disk (deleted worktree)
        let info = worktree_repo_project(wt_path.to_str().expect("utf8"));
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.repo_project, "my-backend");
        assert_eq!(
            info.git_root.as_deref(),
            Some(repo_dir.to_str().expect("utf8"))
        );
        assert_eq!(
            info.git_common_dir.as_deref(),
            Some(repo_dir.join(".git").to_str().expect("utf8"))
        );

        let git = git_metadata_for_cwd(wt_path.to_str().expect("utf8"));
        assert_eq!(git.repo_project.as_deref(), Some("my-backend"));
        assert_eq!(git.status, "path-fallback");
    }

    #[test]
    fn worktree_repo_info_rejects_sibling_without_surviving_repo() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let wt_path = tmp.path().join("missing-backend.wt-hotfix");
        let info = worktree_repo_project(wt_path.to_str().expect("utf8"));
        assert_eq!(info, None);
    }

    #[test]
    fn opencode_cwd_resolution_from_database() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("opencode.db");
        let conn = Connection::open(&db_path).expect("open db");
        conn.execute_batch(
            "CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT, name TEXT);
             CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, directory TEXT, title TEXT, time_created INTEGER);
             INSERT INTO project (id, worktree, name) VALUES ('p1', '/repo/project-wt', 'project-name');
             INSERT INTO session (id, project_id, parent_id, directory, title, time_created)
                 VALUES ('s-dir', 'p1', NULL, '/repo/session-dir', 'Session Dir', 1000);
             INSERT INTO session (id, project_id, parent_id, directory, title, time_created)
                 VALUES ('s-wt', 'p1', NULL, NULL, 'Session WT', 2000);
             INSERT INTO session (id, project_id, parent_id, directory, title, time_created)
                 VALUES ('s-none', NULL, NULL, NULL, 'Session None', 3000);",
        )
        .expect("setup db");

        let mut cache = OpencodeLookupCache::default();
        let db_str = db_path.to_str().expect("utf8");

        assert_eq!(
            resolve_session_cwd_from_parts(SourceKind::Opencode, db_str, "s-dir", &mut cache)
                .as_deref(),
            Some("/repo/session-dir")
        );
        assert_eq!(
            resolve_session_cwd_from_parts(SourceKind::Opencode, db_str, "s-wt", &mut cache)
                .as_deref(),
            Some("/repo/project-wt")
        );
        assert_eq!(
            resolve_session_cwd_from_parts(SourceKind::Opencode, db_str, "s-none", &mut cache)
                .as_deref(),
            None
        );
    }

    #[test]
    fn omp_and_cursor_cwd_resolution() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut cache = OpencodeLookupCache::default();

        // OMP session with "id" and "cwd"
        let omp_file = tmp.path().join("omp_session.jsonl");
        fs::write(
            &omp_file,
            "{\"type\":\"session\",\"version\":3,\"id\":\"omp-123\",\"cwd\":\"/repo/omp-app\"}\n",
        )
        .expect("write omp");
        let omp_cwd = resolve_session_cwd_from_parts(
            SourceKind::Omp,
            omp_file.to_str().expect("utf8"),
            "omp-123",
            &mut cache,
        );
        assert_eq!(omp_cwd.as_deref(), Some("/repo/omp-app"));

        // Cursor session with existing directory
        let cursor_dir = tmp.path().join("cursor-repo");
        fs::create_dir_all(&cursor_dir).expect("mkdir cursor repo");
        let cursor_file = tmp.path().join("cursor_session.json");
        fs::write(
            &cursor_file,
            format!("{{\"cwd\":\"{}\"}}\n", cursor_dir.display()),
        )
        .expect("write cursor");
        let cursor_cwd = resolve_session_cwd_from_parts(
            SourceKind::Cursor,
            cursor_file.to_str().expect("utf8"),
            "cursor-123",
            &mut cache,
        );
        assert_eq!(
            cursor_cwd.as_deref(),
            Some(cursor_dir.to_str().expect("utf8"))
        );
    }

    #[test]
    fn worktree_repo_info_rejects_non_git_sibling() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let plain_dir = tmp.path().join("plain-folder");
        fs::create_dir_all(&plain_dir).expect("mkdir plain dir");
        // No .git directory in plain_dir!

        let wt_path = tmp.path().join("plain-folder-wt-2024");
        let info = worktree_repo_project(wt_path.to_str().expect("utf8"));
        assert_eq!(info, None);
    }

    #[test]
    fn cursor_cwd_resolution_supports_deleted_worktree() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut cache = OpencodeLookupCache::default();
        let non_existent_dir = tmp.path().join("deleted-repo.wt-hotfix");
        let cursor_file = tmp.path().join("cursor_session.json");
        fs::write(
            &cursor_file,
            format!("{{\"cwd\":\"{}\"}}\n", non_existent_dir.display()),
        )
        .expect("write cursor");

        let cursor_cwd = resolve_session_cwd_from_parts(
            SourceKind::Cursor,
            cursor_file.to_str().expect("utf8"),
            "cursor-deleted",
            &mut cache,
        );
        assert_eq!(
            cursor_cwd.as_deref(),
            Some(non_existent_dir.to_str().expect("utf8"))
        );
    }

    #[test]
    fn worktree_repo_info_resolves_dot_delimited_sibling() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo = tmp.path().join("BenchBox");
        fs::create_dir_all(repo.join(".git")).expect("mkdir repo .git");

        let wt_path = tmp.path().join("BenchBox.pool-01");
        let info = worktree_repo_project(wt_path.to_str().expect("utf8"));
        assert!(info.is_some());
        assert_eq!(info.unwrap().repo_project, "BenchBox");
    }

    #[test]
    fn worktree_repo_info_resolves_container_worktree() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo = tmp.path().join("omnigent");
        fs::create_dir_all(repo.join(".git")).expect("mkdir repo .git");

        let wt_path = tmp
            .path()
            .join("omnigent-worktrees")
            .join("muse-production");
        let info = worktree_repo_project(wt_path.to_str().expect("utf8"));
        assert!(info.is_some());
        assert_eq!(info.unwrap().repo_project, "omnigent");
    }

    #[test]
    fn claude_cwd_from_source_path_resolves_encoded_repo() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let claude_projects = tmp.path().join(".claude").join("projects");
        let session_file = claude_projects
            .join("-Users-joe-Developer-BenchBox")
            .join("sess-1.jsonl");

        let cwd = claude_cwd_from_source_path(session_file.to_str().expect("utf8"));
        assert_eq!(cwd.as_deref(), Some("/Users/joe/Developer/BenchBox"));

        let tmp_session = claude_projects
            .join("-tmp-sandbox-repo")
            .join("sess-2.jsonl");
        let cwd = claude_cwd_from_source_path(tmp_session.to_str().expect("utf8"));
        assert_eq!(cwd.as_deref(), Some("/tmp/sandbox-repo"));

        let priv_session = claude_projects
            .join("-private-tmp-worker-test")
            .join("sess-3.jsonl");
        let cwd = claude_cwd_from_source_path(priv_session.to_str().expect("utf8"));
        assert_eq!(cwd.as_deref(), Some("/private/tmp/worker-test"));
    }

    #[test]
    fn worktree_repo_info_resolves_tmp_worker_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dev_dir = tmp.path().join("Developer");
        let repo = dev_dir.join("my-service");
        fs::create_dir_all(repo.join(".git")).expect("mkdir repo .git");
        let _guard = crate::test_support::EnvVarGuard::set(&[(
            "HOME",
            Some(tmp.path().to_str().expect("utf8")),
        )]);

        let worker_dir = Path::new("/tmp/my-service-worker-task-123/sub");
        let info = worktree_repo_project(worker_dir.to_str().expect("utf8"));
        assert!(info.is_some());
        assert_eq!(info.unwrap().repo_project, "my-service");

        let dot_worker = Path::new("/private/tmp/my-service.worker-456");
        let info = worktree_repo_project(dot_worker.to_str().expect("utf8"));
        assert!(info.is_some());
        assert_eq!(info.unwrap().repo_project, "my-service");
    }

    #[test]
    fn sanitize_label_handles_unclosed_tags_safely() {
        assert_eq!(
            sanitize_label("Valid prompt <system-reminder unclosed"),
            "Valid prompt"
        );
        assert_eq!(
            sanitize_label("Valid prompt <system-reminder>never closed content"),
            "Valid prompt"
        );
        assert_eq!(
            sanitize_label("Valid prompt <command-name>echo hi</command-name> trailing"),
            "Valid prompt trailing"
        );
    }

    #[test]
    fn schema_9_migration_recomputes_repo_project() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("analytics.sqlite");
        let dev_dir = tmp.path().join("Developer");
        let repo_dir = dev_dir.join("MyCoolRepo");
        fs::create_dir_all(repo_dir.join(".git")).expect("mkdir repo");

        {
            let conn = Connection::open(&db_path).expect("open sqlite");
            conn.execute_batch(
                r#"
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO meta(key, value) VALUES('schema_version', '6');
                INSERT INTO meta(key, value) VALUES('analytics_complete', '1');
                CREATE TABLE sessions (
                    source TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    project TEXT NOT NULL,
                    cwd TEXT,
                    git_root TEXT,
                    git_common_dir TEXT,
                    repo_project TEXT,
                    started_at INTEGER NOT NULL,
                    last_at INTEGER NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    resolution_status TEXT NOT NULL DEFAULT '',
                    label TEXT,
                    conversation_kind TEXT,
                    PRIMARY KEY (source, session_id, source_path)
                );
                "#,
            )
            .expect("seed schema 6");

            let wt_path = dev_dir.join("MyCoolRepo.pool-02");
            conn.execute(
                "INSERT INTO sessions (source, session_id, source_path, project, cwd, started_at, last_at)
                 VALUES ('codex', 's-wt1', '/tmp/s.jsonl', 'MyCoolRepo.pool-02', ?1, 1000, 2000)",
                params![wt_path.to_str().expect("utf8")],
            )
            .expect("insert session");

            let claude_source = dev_dir
                .join(".claude")
                .join("projects")
                .join("-Users-joe-Developer-memex")
                .join("session.jsonl");
            conn.execute(
                "INSERT INTO sessions (source, session_id, source_path, project, cwd, started_at, last_at)
                 VALUES ('claude', 's-claude1', ?1, '-Users-joe-Developer-memex', NULL, 1000, 2000)",
                params![claude_source.to_str().expect("utf8")],
            )
            .expect("insert claude session");
        }

        // Opening store triggers upgrade to schema 9 and runs migrate_v9_repo_projects
        let store = AnalyticsStore::open(&db_path).expect("open and migrate");
        let repo_proj: Option<String> = store
            .conn
            .query_row(
                "SELECT repo_project FROM sessions WHERE session_id = 's-wt1'",
                [],
                |row| row.get(0),
            )
            .expect("query migrated repo_project");
        assert_eq!(repo_proj.as_deref(), Some("MyCoolRepo"));

        let claude_proj: Option<String> = store
            .conn
            .query_row(
                "SELECT repo_project FROM sessions WHERE session_id = 's-claude1'",
                [],
                |row| row.get(0),
            )
            .expect("query migrated claude repo_project");
        assert_eq!(claude_proj.as_deref(), Some("memex"));
    }
}
