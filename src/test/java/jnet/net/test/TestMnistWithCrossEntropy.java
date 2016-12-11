package jnet.net.test;

import static org.junit.Assert.*;

import java.io.InputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import jnet.algorithm.LearningAlgorithm;
import jnet.algorithm.StochasticGradientDescent;
import jnet.data.DataSet;
import jnet.data.DataSetLoader;
import jnet.net.CostFunction;
import jnet.net.CrossEntropyCostFunction;
import jnet.net.FeedForwardNetwork;
import jnet.net.NetworkException;
import jnet.net.QuadraticCostFunction;
import jnet.net.SigmoidFunction;

public class TestMnistWithCrossEntropy {

	private String trainingSetZip = "./src/test/resources/mnist_train.csv.zip";
	private String trainingSetFileName = "mnist_train.csv";
	private String testSetFileName = "./src/test/resources/mnist_test.csv";
	private DataSet trainingSet;
	private DataSet testSet;
	
	@Before
	public void setUp() throws Exception {
		try (ZipFile zipFile = new ZipFile(trainingSetZip)) {
			ZipEntry entry = zipFile.getEntry(trainingSetFileName);
			InputStream stream = zipFile.getInputStream(entry);
			trainingSet = DataSetLoader.loadFromInputStream(stream, "csv", 10, 5.0/6.0, 1.0/6.0);
			trainingSet.normalize();
			testSet = DataSetLoader.loadFromFile(testSetFileName, "csv", 10, 1.0, 0.0);
			testSet.normalize();
		}
	}

	@After
	public void tearDown() throws Exception {
	}
	
	@Test
	public void testMnist() {
		FeedForwardNetwork network = new FeedForwardNetwork(new int[] {784, 30, 10}, new SigmoidFunction());
		CostFunction costFunction = new CrossEntropyCostFunction();
		int numEpochs = 30;
		double learningRate = 2.0;
		double momentum = 0.5;
		int batchSize = 10;
		LearningAlgorithm sgd = new StochasticGradientDescent(numEpochs, batchSize, learningRate, momentum);
		try {
			network.validateOrTest(testSet, costFunction).print(false);
			network.train(trainingSet.getTrainingSubset(), trainingSet.getValidationSubset(), sgd, costFunction);
			network.validateOrTest(testSet, costFunction).print(true);
		} catch (NetworkException e) {
			assertTrue("Test failed", false);
			e.printStackTrace();
		} catch (RuntimeException e) {
			assertTrue("Test failed", false);
			e.printStackTrace();
		}
	}

}
