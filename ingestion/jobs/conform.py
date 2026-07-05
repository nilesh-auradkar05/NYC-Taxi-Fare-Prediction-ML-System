"""Glue PySpark conform job for NYC Mobility Platform T-101.

Scope:
- Yellow first, with service mapping for later HVFHV extension.
- Cast/normalize core trip fields.
- Deterministic trip_id.
- Zone lookup left joins with quarantine on missing zone mapping.
- Derived avg_mph, fare_ex_surcharge, tip_rate.
- Iceberg MERGE into silver.trips and silver.trips_quarantine.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

NUMERIC_RE = r"^-?\d+(\.\d+)?$"


@dataclass(frozen=True)
class ServiceSpec:
    pickup_ts: str
    dropoff_ts: str
    pu_zone_id: str
    do_zone_id: str
    distance: str
    fare: str
    tip: str
    total: str
    vendor_id: str | None = None
    passenger_count: str | None = None
    extra: str | None = None
    mta_tax: str | None = None
    tolls: str | None = None
    improvement_surcharge: str | None = None
    congestion_surcharge: str | None = None
    airport_fee: str | None = None
    cbd_congestion_fee: str | None = None


SERVICE_SPECS: dict[str, ServiceSpec] = {
    "yellow": ServiceSpec(
        pickup_ts="tpep_pickup_datetime",
        dropoff_ts="tpep_dropoff_datetime",
        pu_zone_id="PULocationID",
        do_zone_id="DOLocationID",
        distance="trip_distance",
        fare="fare_amount",
        tip="tip_amount",
        total="total_amount",
        vendor_id="VendorID",
        passenger_count="passenger_count",
        extra="extra",
        mta_tax="mta_tax",
        tolls="tolls_amount",
        improvement_surcharge="improvement_surcharge",
        congestion_surcharge="congestion_surcharge",
        airport_fee="Airport_fee",
        cbd_congestion_fee="cbd_congestion_fee",
    ),
    # Not enabled in T-101 runs, but the mapping shape is ready for T-106/HVFHV.
    "hvfhv": ServiceSpec(
        pickup_ts="pickup_datetime",
        dropoff_ts="dropoff_datetime",
        pu_zone_id="PULocationID",
        do_zone_id="DOLocationID",
        distance="trip_miles",
        fare="base_passenger_fare",
        tip="tips",
        total="base_passenger_fare",
        vendor_id="hvfhs_license_num",
        passenger_count=None,
        tolls="tolls",
        congestion_surcharge="congestion_surcharge",
        airport_fee="airport_fee",
        cbd_congestion_fee="cbd_congestion_fee",
    ),
}


SILVER_TRIPS_COLUMNS = [
    "trip_id",
    "service",
    "year_month",
    "source_file",
    "vendor_id",
    "passenger_count",
    "pickup_ts",
    "dropoff_ts",
    "pu_zone_id",
    "do_zone_id",
    "pu_borough",
    "pu_zone",
    "pu_service_zone",
    "do_borough",
    "do_zone",
    "do_service_zone",
    "trip_distance",
    "duration_s",
    "avg_mph",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
    "total_amount",
    "fare_ex_surcharge",
    "tip_rate",
]

QUARANTINE_COLUMNS = [
    "quarantine_id",
    "trip_id",
    "reason_code",
    "source_file",
    "service",
    "year_month",
    "pickup_ts",
    "dropoff_ts",
    "pu_zone_id",
    "do_zone_id",
    "raw_record_json",
]


def _column_or_null(df: DataFrame, name: str | None, dtype: T.DataType | str) -> F.Column:
    if name and name in df.columns:
        return F.col(name).cast(dtype)
    return F.lit(None).cast(dtype)


def _safe_decimal(
    df: DataFrame,
    name: str | None,
    precision: int = 14,
    scale: int = 4,
    default_zero: bool = False,
) -> F.Column:
    dtype = T.DecimalType(precision, scale)
    if not name or name not in df.columns:
        if default_zero:
            return F.lit(0).cast(dtype)
        return F.lit(None).cast(dtype)

    value = F.trim(F.col(name).cast("string"))
    parsed = F.when(value.rlike(NUMERIC_RE), value.cast(dtype)).otherwise(F.lit(None).cast(dtype))
    if default_zero:
        return F.coalesce(parsed, F.lit(0).cast(dtype))
    return parsed


def _source_file_col(df: DataFrame) -> F.Column:
    if "source_file" in df.columns:
        return F.coalesce(F.col("source_file"), F.input_file_name())
    return F.input_file_name()


def _raw_record_json(df: DataFrame) -> F.Column:
    fields = [F.col(c).cast("string").alias(c) for c in df.columns]
    return F.to_json(F.struct(*fields))


def normalize_trips(raw_df: DataFrame, service: str, year_month: str) -> DataFrame:
    service = service.lower()
    if service not in SERVICE_SPECS:
        raise ValueError(f"Unsupported service for T-101 conform job: {service}")

    spec = SERVICE_SPECS[service]
    df = raw_df.withColumn("_raw_record_json", _raw_record_json(raw_df))

    normalized = df.select(
        F.lit(service).alias("service"),
        F.lit(year_month).alias("year_month"),
        _source_file_col(df).alias("source_file"),
        _column_or_null(df, spec.vendor_id, "string").alias("vendor_id"),
        _column_or_null(df, spec.passenger_count, "int").alias("passenger_count"),
        F.to_timestamp(_column_or_null(df, spec.pickup_ts, "string")).alias("pickup_ts"),
        F.to_timestamp(_column_or_null(df, spec.dropoff_ts, "string")).alias("dropoff_ts"),
        _column_or_null(df, spec.pu_zone_id, "int").alias("pu_zone_id"),
        _column_or_null(df, spec.do_zone_id, "int").alias("do_zone_id"),
        _safe_decimal(df, spec.distance, precision=12, scale=4).alias("trip_distance"),
        _safe_decimal(df, spec.fare, precision=14, scale=4).alias("fare_amount"),
        _safe_decimal(df, spec.extra, default_zero=True).alias("extra"),
        _safe_decimal(df, spec.mta_tax, default_zero=True).alias("mta_tax"),
        _safe_decimal(df, spec.tip, default_zero=True).alias("tip_amount"),
        _safe_decimal(df, spec.tolls, default_zero=True).alias("tolls_amount"),
        _safe_decimal(df, spec.improvement_surcharge, default_zero=True)
        .alias("improvement_surcharge"),
        _safe_decimal(df, spec.congestion_surcharge, default_zero=True)
        .alias("congestion_surcharge"),
        _safe_decimal(df, spec.airport_fee, default_zero=True).alias("airport_fee"),
        _safe_decimal(df, spec.cbd_congestion_fee, default_zero=True).alias("cbd_congestion_fee"),
        _safe_decimal(df, spec.total, precision=14, scale=4).alias("total_amount"),
        F.col("_raw_record_json").alias("raw_record_json"),
    )

    normalized = normalized.withColumn(
        "duration_s",
        F.unix_timestamp("dropoff_ts") - F.unix_timestamp("pickup_ts"),
    )

    normalized = normalized.withColumn(
        "avg_mph",
        F.when(
            (F.col("trip_distance") > 0) & (F.col("duration_s") > 0),
            F.round(F.col("trip_distance").cast("double") * F.lit(3600.0) / F.col("duration_s"), 4),
        ).otherwise(F.lit(None).cast("double")),
    )

    normalized = normalized.withColumn(
        "fare_ex_surcharge",
        F.col("fare_amount").cast(T.DecimalType(14, 4)),
    )

    normalized = normalized.withColumn(
        "tip_rate",
        F.when(
            F.col("fare_ex_surcharge") > 0,
            F.round(F.col("tip_amount").cast("double") / F.col("fare_ex_surcharge").cast("double"),
                    6),
        ).otherwise(F.lit(None).cast("double")),
    )

    return add_trip_id(normalized)


def add_trip_id(df: DataFrame) -> DataFrame:
    key_columns = [
        "service",
        "vendor_id",
        "pickup_ts",
        "dropoff_ts",
        "pu_zone_id",
        "do_zone_id",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "total_amount",
    ]
    canonical = [
        F.coalesce(F.col(c).cast("string"), F.lit(""))
        for c in key_columns
    ]
    return df.withColumn("trip_id", F.sha2(F.concat_ws("||", *canonical), 256))


def normalize_zone_lookup(zone_df: DataFrame) -> DataFrame:
    return (
        zone_df.select(
            F.col("LocationID").cast("int").alias("zone_id"),
            F.col("Borough").cast("string").alias("borough"),
            F.col("Zone").cast("string").alias("zone"),
            F.col("service_zone").cast("string").alias("service_zone"),
        )
        .where(F.col("zone_id").isNotNull())
        .dropDuplicates(["zone_id"])
    )


def join_zones(trips_df: DataFrame, zone_df: DataFrame) -> DataFrame:
    zones = normalize_zone_lookup(zone_df)

    joined = (
        trips_df.alias("t")
        .join(zones.alias("pu"), F.col("t.pu_zone_id") == F.col("pu.zone_id"), "left")
        .join(zones.alias("do"), F.col("t.do_zone_id") == F.col("do.zone_id"), "left")
    )

    return joined.select(
        "t.*",
        F.col("pu.borough").alias("pu_borough"),
        F.col("pu.zone").alias("pu_zone"),
        F.col("pu.service_zone").alias("pu_service_zone"),
        F.col("do.borough").alias("do_borough"),
        F.col("do.zone").alias("do_zone"),
        F.col("do.service_zone").alias("do_service_zone"),
    )


def classify_records(joined_df: DataFrame) -> DataFrame:
    reason_exprs = [
        F.when(F.col("pickup_ts").isNull(), F.lit("BAD_PICKUP_TS")),
        F.when(F.col("dropoff_ts").isNull(), F.lit("BAD_DROPOFF_TS")),
        F.when((F.col("duration_s").isNull()) | (F.col("duration_s") <= 0), F.lit("BAD_DURATION")),
        F.when((F.col("trip_distance").isNull()) | (F.col("trip_distance") <= 0),
               F.lit("BAD_DISTANCE")),
        F.when(F.col("fare_amount").isNull(), F.lit("BAD_FARE")),
        F.when((F.col("pu_zone_id").isNull()) | F.col("pu_borough").isNull(), F.lit("BAD_PU_ZONE")),
        F.when((F.col("do_zone_id").isNull()) | F.col("do_borough").isNull(), F.lit("BAD_DO_ZONE")),
        F.when(
            F.date_format(F.col("pickup_ts"), "yyyy-MM") != F.col("year_month"),
            F.lit("MONTH_MISMATCH"),
        ),
    ]

    return (
        joined_df.withColumn("reason_codes", F.array(*reason_exprs))
        .withColumn("reason_codes", F.expr("filter(reason_codes, x -> x is not null)"))
        .withColumn("reason_code", F.concat_ws(",", F.col("reason_codes")))
    )


def split_valid_and_quarantine(classified_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    valid = classified_df.where(F.size("reason_codes") == 0)

    window = Window.partitionBy("trip_id").orderBy(F.col("source_file").asc_nulls_last())
    valid = (
        valid.withColumn("_rn", F.row_number().over(window))
        .where(F.col("_rn") == 1)
        .drop("_rn", "reason_codes", "reason_code", "raw_record_json")
        .select(*SILVER_TRIPS_COLUMNS)
    )

    rejected = classified_df.where(F.size("reason_codes") > 0)
    rejected = (
        rejected.withColumn(
            "quarantine_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.coalesce(F.col("trip_id"), F.lit("")),
                    F.col("reason_code"),
                    F.coalesce(F.col("source_file"), F.lit("")),
                    F.col("service"),
                    F.col("year_month"),
                    F.coalesce(F.col("raw_record_json"), F.lit("")),
                ),
                256,
            ),
        )
        .select(*QUARANTINE_COLUMNS)
        .dropDuplicates(["quarantine_id"])
    )

    return valid, rejected


def conform_trips(raw_df: DataFrame, zone_df: DataFrame, service: str, year_month: str) -> tuple[
    DataFrame, DataFrame]:
        normalized = normalize_trips(raw_df, service=service, year_month=year_month)
        joined = join_zones(normalized, zone_df)
        classified = classify_records(joined)
        return split_valid_and_quarantine(classified)


def load_zone_lookup(spark: SparkSession, zone_lookup_path: str) -> DataFrame:
    return (
        spark.read.option("header", "true")
        .option("mode", "FAILFAST")
        .csv(zone_lookup_path)
    )


def ensure_iceberg_tables(spark: SparkSession, catalog: str, silver_db: str) -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {catalog}.{silver_db}.trips (
            trip_id string,
            service string,
            year_month string,
            source_file string,
            vendor_id string,
            passenger_count int,
            pickup_ts timestamp,
            dropoff_ts timestamp,
            pu_zone_id int,
            do_zone_id int,
            pu_borough string,
            pu_zone string,
            pu_service_zone string,
            do_borough string,
            do_zone string,
            do_service_zone string,
            trip_distance decimal(12,4),
            duration_s bigint,
            avg_mph double,
            fare_amount decimal(14,4),
            extra decimal(14,4),
            mta_tax decimal(14,4),
            tip_amount decimal(14,4),
            tolls_amount decimal(14,4),
            improvement_surcharge decimal(14,4),
            congestion_surcharge decimal(14,4),
            airport_fee decimal(14,4),
            cbd_congestion_fee decimal(14,4),
            total_amount decimal(14,4),
            fare_ex_surcharge decimal(14,4),
            tip_rate double
        )
        USING iceberg
        PARTITIONED BY (months(pickup_ts))
        TBLPROPERTIES ('format-version'='2')
        """
    )

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {catalog}.{silver_db}.trips_quarantine (
            quarantine_id string,
            trip_id string,
            reason_code string,
            source_file string,
            service string,
            year_month string,
            pickup_ts timestamp,
            dropoff_ts timestamp,
            pu_zone_id int,
            do_zone_id int,
            raw_record_json string
        )
        USING iceberg
        PARTITIONED BY (year_month, service)
        TBLPROPERTIES ('format-version'='2')
        """
    )

def _quote_identifier(name: str) -> str:
    return f"`{name}`"


def _temp_path(temp_base_path: str, table_name: str) -> str:
    return f"{temp_base_path.rstrip('/')}/{table_name}/{uuid.uuid4().hex}"


def materialize_for_merge(
    spark: SparkSession,
    df: DataFrame,
    columns: Iterable[str],
    temp_base_path: str,
    table_name: str,
) -> DataFrame:
    """Write/read staging data to cut non-deterministic Spark lineage before MERGE.

    Glue/Spark can reject MERGE sources whose logical plan still contains
    non-deterministic expressions such as input_file_name(). Materializing the
    source to Parquet turns the merge input into plain attributes.
    """
    cols = list(columns)
    staged_df = df.select(*cols)

    if staged_df.rdd.isEmpty():
        return staged_df

    path = _temp_path(temp_base_path=temp_base_path, table_name=table_name)
    staged_df.write.mode("overwrite").parquet(path)
    return spark.read.parquet(path).select(*cols)

def merge_into_iceberg(
    spark: SparkSession,
    df: DataFrame,
    target_table: str,
    columns: Iterable[str],
    key_column: str,
    staging_view: str,
) -> None:
    cols = list(columns)
    if df.rdd.isEmpty():
        return

    df.select(*cols).createOrReplaceTempView(staging_view)

    update_clause = ",\n                ".join(
        f"t.{_quote_identifier(c)} = s.{_quote_identifier(c)}"
        for c in cols
        if c != key_column
    )
    insert_cols = ", ".join(_quote_identifier(c) for c in cols)
    insert_vals = ", ".join(f"s.{_quote_identifier(c)}" for c in cols)

    spark.sql(
        f"""
        MERGE INTO {target_table} t
        USING (SELECT {insert_cols} FROM {staging_view}) s
        ON t.{_quote_identifier(key_column)} = s.{_quote_identifier(key_column)}
        WHEN MATCHED THEN UPDATE SET
                {update_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_cols})
        VALUES ({insert_vals})
        """
    )


def write_outputs(
    spark: SparkSession,
    valid_df: DataFrame,
    quarantine_df: DataFrame,
    catalog: str,
    silver_db: str,
    temp_base_path: str,
) -> None:
    ensure_iceberg_tables(spark, catalog=catalog, silver_db=silver_db)

    staged_valid_df = materialize_for_merge(
        spark=spark,
        df=valid_df,
        columns=SILVER_TRIPS_COLUMNS,
        temp_base_path=temp_base_path,
        table_name="trips",
    )

    staged_quarantine_df = materialize_for_merge(
        spark=spark,
        df=quarantine_df,
        columns=QUARANTINE_COLUMNS,
        temp_base_path=temp_base_path,
        table_name="trips_quarantine",
    )

    merge_into_iceberg(
        spark=spark,
        df=staged_valid_df,
        target_table=f"{catalog}.{silver_db}.trips",
        columns=SILVER_TRIPS_COLUMNS,
        key_column="trip_id",
        staging_view="staged_trips",
    )

    merge_into_iceberg(
        spark=spark,
        df=staged_quarantine_df,
        target_table=f"{catalog}.{silver_db}.trips_quarantine",
        columns=QUARANTINE_COLUMNS,
        key_column="quarantine_id",
        staging_view="staged_trips_quarantine",
    )


def _optional_arg(argv: list[str], name: str, default: str) -> str:
    flag = f"--{name}"
    if flag not in argv:
        return default

    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"Missing value for optional argument {flag}")

    return argv[index + 1]


def parse_args(argv: list[str]) -> dict[str, str]:
    arg_names = [
        "JOB_NAME",
        "service",
        "year_month",
        "bronze_base_path",
        "zone_lookup_path",
        "catalog",
        "silver_db",
    ]

    try:
        from awsglue.utils import getResolvedOptions  # type: ignore

        args = getResolvedOptions(argv, arg_names)
    except ImportError:
        parser = argparse.ArgumentParser()
        for arg in arg_names:
            if arg == "JOB_NAME":
                continue
            parser.add_argument(f"--{arg}", required=True)
        parser.add_argument("--temp_base_path", required=False)
        args = vars(parser.parse_args(argv[1:]))

    if not args.get("temp_base_path"):
        args["temp_base_path"] = _optional_arg(
            argv,
            "temp_base_path",
            f"{args['bronze_base_path'].rstrip('/')}/_tmp/conform_merge_staging",
        )

    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    service = args["service"].lower()
    year_month = args["year_month"]

    spark = SparkSession.builder.appName(
        f"nyc-mobility-conform-{service}-{year_month}").getOrCreate()

    bronze_path = (
        f"{args['bronze_base_path'].rstrip('/')}/"
        f"service={service}/year_month={year_month}/"
    )

    raw_df = (
        spark.read.option("pathGlobFilter", "*.parquet")
        .parquet(bronze_path)
        .withColumn("source_file", F.input_file_name())
    )

    zone_df = load_zone_lookup(spark, args["zone_lookup_path"])
    valid_df, quarantine_df = conform_trips(
        raw_df=raw_df,
        zone_df=zone_df,
        service=service,
        year_month=year_month,
    )

    write_outputs(
        spark=spark,
        valid_df=valid_df,
        quarantine_df=quarantine_df,
        catalog=args["catalog"],
        silver_db=args["silver_db"],
        temp_base_path=args["temp_base_path"],
    )


if __name__ == "__main__":
    main(sys.argv)
