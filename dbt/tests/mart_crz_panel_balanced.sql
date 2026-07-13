with daily_outcomes as (
    select pickup_date, pu_zone_id
    from {{ ref('fct_trips_zone_day') }}
),

date_bounds as (
    select
        cast(date_trunc('week', min(pickup_date)) as date) as min_week,
        cast(date_trunc('week', max(pickup_date)) as date) as max_week
    from daily_outcomes
),

{% if target.type == 'athena' %}
date_spine as (
    select week_start
    from date_bounds
    cross join unnest(sequence(min_week, max_week, interval '7' day)) as weeks (week_start)
),
{% else %}
date_spine as (
    select cast(week_start as date) as week_start
    from date_bounds
    cross join generate_series(min_week, max_week, interval '7' day) as weeks (week_start)
),
{% endif %}

zone_spine as (
    select distinct pu_zone_id
    from daily_outcomes

    union

    select cast(location_id as integer) as pu_zone_id
    from {{ ref('crz_zones') }}

    union

    select cast(location_id as integer) as pu_zone_id
    from {{ ref('buffer_ring') }}
),

expected as (
    select zone_spine.pu_zone_id, date_spine.week_start
    from zone_spine
    cross join date_spine
),

actual as (
    select pu_zone_id, week_start
    from {{ ref('mart_crz_panel') }}
)

select
    coalesce(expected.pu_zone_id, actual.pu_zone_id) as pu_zone_id,
    coalesce(expected.week_start, actual.week_start) as week_start
from expected
full outer join actual
    on expected.pu_zone_id = actual.pu_zone_id
    and expected.week_start = actual.week_start
where expected.pu_zone_id is null
    or actual.pu_zone_id is null
