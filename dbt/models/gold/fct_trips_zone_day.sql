{% if target.type == 'athena' %}
    {{ config(
        materialized='table',
        table_type='iceberg',
        format='parquet',
        contract={'enforced': true}
    ) }}
{% else %}
    {{ config(
        materialized='table',
        contract={'enforced': true}
    ) }}
{% endif %}

with source_trips as (
    select
        cast(pickup_ts as date) as pickup_date,
        cast(pu_zone_id as integer) as pu_zone_id,
        cast(service as varchar) as service,
        cast(total_amount as decimal(18, 4)) as total_amount,
        cast(fare_ex_surcharge as double) as fare_ex_surcharge,
        cast(tip_rate as double) as tip_rate,
        cast(avg_mph as double) as avg_mph,
        cast(cbd_congestion_fee as decimal(18, 4)) as cbd_congestion_fee
    from {{ source('silver', 'trips') }}
)

select
    pickup_date,
    pu_zone_id,
    service,
    cast(count(*) as bigint) as trips,
    cast(sum(total_amount) as decimal(18, 4)) as total_fare,
    cast(avg(fare_ex_surcharge) as double) as avg_fare_ex_surcharge,
    cast(avg(tip_rate) as double) as tip_rate,
    cast(avg(avg_mph) as double) as avg_mph,
    cast(sum(cbd_congestion_fee) as decimal(18, 4)) as cbd_fee_revenue
from source_trips
group by
    pickup_date,
    pu_zone_id,
    service
