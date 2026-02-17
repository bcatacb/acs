# Asset Data Model (V1 Scaffold)

SQLite file: `backend/assets/acs.db`

## Tables

### `assets`
- `id` TEXT PK
- `path` TEXT UNIQUE
- `type` TEXT (`drum`, `loop`, `midi`, `audio`)
- `role` TEXT (`kick`, `hat`, `clap`, `chords`, `melodies`, `basslines`, `drums`, etc.)
- `bpm` INTEGER nullable
- `musical_key` TEXT nullable
- `style_tag` TEXT nullable
- `energy` TEXT nullable
- `tags_json` TEXT JSON array
- `active` INTEGER default 1
- `created_at` TEXT
- `updated_at` TEXT

### `profiles`
- `id` TEXT PK
- `name` TEXT UNIQUE
- `json_config` TEXT JSON object
- `tags_json` TEXT JSON array
- `created_at` TEXT

### `generations`
- `id` TEXT PK
- `user_id` TEXT nullable
- `project_id` TEXT nullable
- `profile_id` TEXT nullable
- `asset_ids_json` TEXT JSON array
- `params_json` TEXT JSON object
- `output_path` TEXT nullable
- `created_at` TEXT

## Ingest Command

From `backend`:

```bash
python scripts/ingest_assets.py ../asset_drop
```

Optional tags:

```bash
python scripts/ingest_assets.py ../asset_drop --style boom_bap --tag dark_fast
```

## Runtime Behavior
- Generator now attempts DB-backed catalog first.
- If DB has no entries, it falls back to existing folder scan logic.
- This keeps current behavior stable while enabling scalable metadata indexing.
