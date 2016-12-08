package jnet.net.test;

import junit.framework.TestCase;
import jnet.net.CostFunction;
import jnet.net.QuadraticCostFunction;
import jnet.net.NetworkException;
import jnet.algorithm.LearningAlgorithm;
import jnet.algorithm.StochasticGradientDescent;
import jnet.data.DataSet;
import jnet.data.DataSetLoader;
import jnet.net.FeedForwardNetwork;

public class TestWine extends TestCase {

	private String dataFileName = "./src/test/resources/wine.csv";
	private DataSet dataSet;
	
	protected void setUp() throws Exception {
		super.setUp();
		dataSet = DataSetLoader.loadFromFile(dataFileName, "csv", "EEEIIIIIIIIIIIII");
		dataSet.normalize();
	}

	protected void tearDown() throws Exception {
		super.tearDown();
	}
	
	public void testWine() {
		//fail("Not yet implemented");
		FeedForwardNetwork network = new FeedForwardNetwork(new int[] {13, 6, 3});
		CostFunction costFunction = new QuadraticCostFunction();
		int numEpochs = 500;
		double learningRate = 0.25;
		int batchSize = 5;
		LearningAlgorithm sgd = new StochasticGradientDescent(numEpochs, batchSize, learningRate, 0);
		try {
			network.train(dataSet, sgd, costFunction);
		} catch (NetworkException e) {
			assertTrue("Test failed", false);
			e.printStackTrace();
		} catch (RuntimeException e) {
			assertTrue("Test failed", false);
			e.printStackTrace();
		}
	}

}
