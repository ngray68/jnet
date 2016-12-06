package jnet.net.test;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import jnet.algorithm.LearningAlgorithm;
import jnet.algorithm.StochasticGradientDescent;
import jnet.data.DataSet;
import jnet.data.DataSetLoader;
import jnet.net.CostFunction;
import jnet.net.FeedForwardNetwork;
import jnet.net.MaxTrueOutputPostProcessor;
import jnet.net.MeanSquaredError;
import jnet.net.NetworkException;

public class TestMnist {

	private String trainingSetFileName = "./src/test/resources/mnist_train.csv";
	private String testSetFileName = "./src/test/resources/mnist_test.csv";
	private DataSet trainingSet;
	private DataSet testSet;
	
	@Before
	public void setUp() throws Exception {
		trainingSet = DataSetLoader.loadFromFile(trainingSetFileName, "csv", 10, 0.9, 0.1);
		trainingSet.normalize();
		testSet = DataSetLoader.loadFromFile(testSetFileName, "csv", 10, 1.0, 0.0);
		testSet.normalize();
	}

	@After
	public void tearDown() throws Exception {
	}
	
	@Test
	public void testMnist() {
		FeedForwardNetwork network = new FeedForwardNetwork(new int[] {784, 30, 10});
		CostFunction costFunction = new MeanSquaredError();
		int numEpochs = 30;
		double learningRate = 5.0;
		int batchSize = 10;
		LearningAlgorithm sgd = new StochasticGradientDescent(numEpochs, batchSize, learningRate, 0);
		try {
			network.validateOrTest(testSet, costFunction).print(false);
			network.train(trainingSet, testSet, sgd, costFunction);
		} catch (NetworkException e) {
			assertTrue("Test failed", false);
			e.printStackTrace();
		} catch (RuntimeException e) {
			assertTrue("Test failed", false);
			e.printStackTrace();
		}
	}


}
