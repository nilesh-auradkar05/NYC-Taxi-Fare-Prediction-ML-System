{% if target.type == 'athena' %}
    {{
        config(
            materialized='table',
            table_type='iceberg',
            format='parquet',
            contract={'enforced': true}
        )
    }}
{% else %}
    {{ config(materialized='table', contract={'enforced': true}) }}
{% endif %}

with daily_outcomes as (
    select
        pickup_date,
        pu_zone_id,
        trips,
        total_fare,
        avg_fare_ex_surcharge,
        tip_rate,
        avg_mph,
        cbd_fee_revenue
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

crz_zones as (
    select distinct cast(location_id as integer) as pu_zone_id
    from {{ ref('crz_zones') }}
),

buffer_ring_zones as (
    select distinct cast(location_id as integer) as pu_zone_id
    from {{ ref('buffer_ring') }}
),

zone_spine as (
    select distinct pu_zone_id
    from daily_outcomes

    union

    select pu_zone_id
    from crz_zones

    union

    select pu_zone_id
    from buffer_ring_zones
),

panel_spine as (
    select
        zone_spine.pu_zone_id,
        date_spine.week_start
    from zone_spine
    cross join date_spine
),

weekly_outcomes as (
    select
        pu_zone_id,
        cast(date_trunc('week', pickup_date) as date) as week_start,
        cast(sum(trips) as bigint) as trips,
        cast(sum(total_fare) as decimal(18, 4)) as total_fare,
        cast(
            sum(
                case
                    when avg_fare_ex_surcharge is not null
                        then avg_fare_ex_surcharge * cast(trips as double)
                    else cast(0 as double)
                end
            )
            / nullif(
                sum(
                    case
                        when avg_fare_ex_surcharge is not null then cast(trips as double)
                        else cast(0 as double)
                    end
                ),
                cast(0 as double)
            )
            as double
        ) as avg_fare_ex_surcharge,
        cast(
            sum(
                case
                    when tip_rate is not null then tip_rate * cast(trips as double)
                    else cast(0 as double)
                end
            )
            / nullif(
                sum(
                    case
                        when tip_rate is not null then cast(trips as double)
                        else cast(0 as double)
                    end
                ),
                cast(0 as double)
            )
            as double
        ) as tip_rate,
        cast(
            sum(
                case
                    when avg_mph is not null then avg_mph * cast(trips as double)
                    else cast(0 as double)
                end
            )
            / nullif(
                sum(
                    case
                        when avg_mph is not null then cast(trips as double)
                        else cast(0 as double)
                    end
                ),
                cast(0 as double)
            )
            as double
        ) as avg_mph,
        cast(sum(cbd_fee_revenue) as decimal(18, 4)) as cbd_fee_revenue
    from daily_outcomes
    group by pu_zone_id, cast(date_trunc('week', pickup_date) as date)
)

select
    cast(panel_spine.pu_zone_id as integer) as pu_zone_id,
    cast(panel_spine.week_start as date) as week_start,
    cast(coalesce(weekly_outcomes.trips, 0) as bigint) as trips,
    cast(coalesce(weekly_outcomes.total_fare, 0) as decimal(18, 4)) as total_fare,
    cast(weekly_outcomes.avg_fare_ex_surcharge as double) as avg_fare_ex_surcharge,
    cast(weekly_outcomes.tip_rate as double) as tip_rate,
    cast(weekly_outcomes.avg_mph as double) as avg_mph,
    cast(coalesce(weekly_outcomes.cbd_fee_revenue, 0) as decimal(18, 4)) as cbd_fee_revenue,
    cast(crz_zones.pu_zone_id is not null as boolean) as is_crz,
    cast(buffer_ring_zones.pu_zone_id is not null as boolean) as is_buffer_ring,
    cast(panel_spine.week_start >= date '2025-01-06' as boolean) as post
from panel_spine
left join weekly_outcomes
    on panel_spine.pu_zone_id = weekly_outcomes.pu_zone_id
    and panel_spine.week_start = weekly_outcomes.week_start
left join crz_zones
    on panel_spine.pu_zone_id = crz_zones.pu_zone_id
left join buffer_ring_zones
    on panel_spine.pu_zone_id = buffer_ring_zones.pu_zone_id
