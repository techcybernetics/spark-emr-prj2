package org.emr;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.io.IOException;


public class RF {

    public static void main(String[] args) throws IOException {

        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = new SparkSession.Builder()
                .appName("Random Forest Model")
                .getOrCreate();
        Dataset<Row> wineQualitydf = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .option("inferSchema", "true")

                .load("s3://cs-643-assignment-2/TrainingDataset.csv");
        wineQualitydf.show(5);

        Dataset<Row> lblFeatureDf = wineQualitydf.withColumnRenamed("\"\"\"\"quality\"\"\"\"\"", "label")
                .withColumnRenamed("\"\"\"\"\"fixed acidity\"\"\"\"", "fixed acidity")
                .withColumnRenamed("\"\"\"\"volatile acidity\"\"\"\"", "volatile acidity")
                .withColumnRenamed("\"\"\"\"citric acid\"\"\"\"", "citric acid")
                .withColumnRenamed("\"\"\"\"residual sugar\"\"\"\"", "residual sugar")
                .withColumnRenamed("\"\"\"\"chlorides\"\"\"\"", "chlorides")
                .withColumnRenamed("\"\"\"\"free sulfur dioxide\"\"\"\"", "free sulfur dioxide")
                .withColumnRenamed("\"\"\"\"total sulfur dioxide\"\"\"\"", "total sulfur dioxide")
                .withColumnRenamed("\"\"\"\"density\"\"\"\"", "density")
                .withColumnRenamed("\"\"\"\"pH\"\"\"\"", "pH")
                .withColumnRenamed("\"\"\"\"sulphates\"\"\"\"", "sulphates")
                .withColumnRenamed("\"\"\"\"alcohol\"\"\"\"", "alcohol")
                .select("label", "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol");

        lblFeatureDf = lblFeatureDf.na().drop();
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"})
                .setOutputCol("features");
        Dataset<Row>[] splitData = lblFeatureDf.randomSplit(new double[]{.7, .3});
        Dataset<Row> trainingDf = splitData[0];
        Dataset<Row> testingDf = splitData[1];
        // Create a Random Forest classifier with 100 trees
        RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(100).setMaxDepth(30);
        Pipeline pl = new Pipeline();
        pl.setStages(new PipelineStage[]{assembler, rf});
        //PipelineModel model = pl.fit(lblFeatureDf);
        PipelineModel model = pl.fit(trainingDf);
        RandomForestClassificationModel rfModel = (RandomForestClassificationModel) model.stages()[1];
        rfModel.save("s3://cs-643-assignment-2/RF.model");
       // rfModel.save("s3://cs-643-assignment-2/RF");
        Dataset<Row> results = model.transform(testingDf);
        results.javaRDD();
        results.show(5);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(results);

        // Print the accuracy
        System.out.println("Accuracy = " + accuracy);


    }


}

