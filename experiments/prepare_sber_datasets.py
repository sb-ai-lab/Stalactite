import os

from rs_datasets import MovieLens
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
from pyspark.sql.types import IntegerType
from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.data.dataset_utils import DatasetLabelEncoder
from replay.metrics import HitRate, NDCG, Experiment
from replay.models import ALSWrap
from replay.utils.spark_utils import convert2spark
from replay.utils.session_handler import State
from replay.splitters import RatioSplitter

GENERATE_FACTORS = False
sbol_path = "/home/dmitriy/data/sber/sbol/"
smm_path = "/home/dmitriy/data/sber/smm/"

if GENERATE_FACTORS:
    spark_sess = (
        SparkSession
        .builder
        .master("local[6]")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.local.dir", "/tmp")
        .getOrCreate()

    )

    spark = State(spark_sess).session
    spark.sparkContext.setLogLevel("ERROR")

    # smm_ds = MovieLens("1m").ratings
    # smm_ds = convert2spark(smm_ds)

    smm_ds = spark.read.parquet(os.path.join(smm_path, "interactions_filtered_117k.parquet"))
    smm_ds = smm_ds.withColumn("rating", sf.lit("1"))
    smm_ds = smm_ds.withColumn("rating", smm_ds.rating.cast(IntegerType()))


    # smm_ds = smm_ds.limit(10000).select("user_id", "item_id", "rating")
    smm_ds = smm_ds.select("user_id", "item_id", "rating")

    print(smm_ds.count())
    # data preprocessing
    # interactions = convert2spark(ml_1m.ratings)

    # # data splitting
    # splitter = RatioSplitter(
    #     test_size=0.3,
    #     divide_column="user_id",
    #     query_column="user_id",
    #     item_column="item_id",
    #     timestamp_column="timestamp",
    #     drop_cold_items=True,
    #     drop_cold_users=True,
    # )
    # train, test = splitter.split(interactions)

    # dataset creating
    feature_schema = FeatureSchema(
        [
            FeatureInfo(
                column="user_id",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.QUERY_ID,
            ),
            FeatureInfo(
                column="item_id",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            FeatureInfo(
                column="rating",
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.RATING,
            ),
            # FeatureInfo(
            #     column="timestamp",
            #     feature_type=FeatureType.NUMERICAL,
            #     feature_hint=FeatureHint.TIMESTAMP,
            # ),
        ]
    )

    train_dataset = Dataset(
        feature_schema=feature_schema,
        interactions=smm_ds,
    )
    # test_dataset = Dataset(
    #     feature_schema=feature_schema,
    #     interactions=test,
    # )

    # data encoding
    # encoder = DatasetLabelEncoder()
    # train_dataset = encoder.fit_transform(train_dataset)
    # test_dataset = encoder.transform(test_dataset)
    # print(train_dataset._interactions.show(5))
    # model training
    model = ALSWrap()
    model.fit(train_dataset)
    fi = train_dataset.query_ids
    factors = model.get_features(train_dataset.query_ids)
    print(factors[0].count())
    # factors[0].show(5)

    factors[0].write.mode("overwrite").parquet(os.path.join(smm_path, "als_user_factors.parquet"))

sbol_labels = pd.read_parquet(os.path.join(sbol_path, "sbol_multilabels.parquet"))
sbol_labels = sbol_labels.iloc[:100]  # sample
sbol_user_features = pd.read_parquet(os.path.join(sbol_path, "user_features.parquet"))
sbol_user_features = sbol_user_features[["user_id", "feature_1", "feature_2", "feature_3"]]

sbol = sbol_labels.merge(sbol_user_features, on="user_id", how="left")

smm_user_factors = pd.read_parquet(os.path.join(smm_path, "als_user_factors.parquet"))
sbol = sbol[sbol["user_id"].isin(smm_user_factors["user_id"].unique())]
print(sbol.shape)

# sbol = sbol.iloc[:1000]
# print(len(smm_user_factors["user_id"].unique()))
# print(smm_user_factors.head())
sbol = sbol.merge(smm_user_factors, on="user_id", how="left")
print(sbol.head())
sbol.to_parquet(os.path.join(sbol_path, "centralised_100_sample.parquet"))
