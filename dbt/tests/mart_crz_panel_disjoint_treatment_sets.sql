select cast(crz.location_id as integer) as pu_zone_id
from {{ ref('crz_zones') }} as crz
inner join {{ ref('buffer_ring') }} as buffer_ring
    on cast(crz.location_id as integer) = cast(buffer_ring.location_id as integer)
