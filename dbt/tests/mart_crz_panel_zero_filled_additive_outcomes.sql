with trip_zone_weeks as (
    select distinct
        pu_zone_id,
        cast(date_trunc('week', pickup_date) as date) as week_start
    from {{ ref('fct_trips_zone_day') }}
)

select
    panel.pu_zone_id,
    panel.week_start,
    panel.trips,
    panel.total_fare,
    panel.cbd_fee_revenue
from {{ ref('mart_crz_panel') }} as panel
left join trip_zone_weeks
    on panel.pu_zone_id = trip_zone_weeks.pu_zone_id
    and panel.week_start = trip_zone_weeks.week_start
where trip_zone_weeks.pu_zone_id is null
    and (
        panel.trips <> 0
        or panel.total_fare <> cast(0 as decimal(18, 4))
        or panel.cbd_fee_revenue <> cast(0 as decimal(18, 4))
    )
