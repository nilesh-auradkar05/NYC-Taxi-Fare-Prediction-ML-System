select
    'trips' as source_table,
    count(*) as fixture_rows
from {{ source('silver', 'trips') }}

union all

select
    'trips_quarantine' as source_table,
    count(*) as fixture_rows
from {{ source('silver', 'trips_quarantine') }}
