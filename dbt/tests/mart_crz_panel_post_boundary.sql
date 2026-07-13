with invalid_flags as (
    select pu_zone_id, week_start
    from {{ ref('mart_crz_panel') }}
    where post <> (week_start >= date '2025-01-06')
),

missing_boundary as (
    select
        cast(null as integer) as pu_zone_id,
        date '2025-01-06' as week_start
    where not exists (
        select 1
        from {{ ref('mart_crz_panel') }}
        where week_start = date '2025-01-06'
            and post
    )
),

missing_pre_boundary as (
    select
        cast(null as integer) as pu_zone_id,
        date '2024-12-30' as week_start
    where not exists (
        select 1
        from {{ ref('mart_crz_panel') }}
        where week_start = date '2024-12-30'
            and not post
    )
)

select pu_zone_id, week_start
from invalid_flags

union all

select pu_zone_id, week_start
from missing_boundary

union all

select pu_zone_id, week_start
from missing_pre_boundary
