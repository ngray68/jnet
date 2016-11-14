package jnet.net.test;

import junit.framework.TestCase;
import jnet.net.CostFunction;
import jnet.net.MeanSquaredError;
import jnet.algorithm.LearningAlgorithm;
import jnet.algorithm.StochasticGradientDescent;
import jnet.data.DataSet;
import jnet.net.MultilayerPerceptronNetwork;

public class TestWine extends TestCase {

	private String dataFileName = "./src/test/resources/wine.data";
	private DataSet dataSet;
	
	protected void setUp() throws Exception {
		super.setUp();
		dataSet = DataSet.createFromFile(dataFileName);
		dataSet.normalize();
	}

	protected void tearDown() throws Exception {
		super.tearDown();
	}
	
	public void testWine() {
		//fail("Not yet implemented");
		MultilayerPerceptronNetwork network = new MultilayerPerceptronNetwork(new int[] {13, 6, 3});
		CostFunction costFunction = new MeanSquaredError();
		int numEpochs = 500;
		double learningRate = 0.25;
		int batchSize = 5;
		LearningAlgorithm sgd = new StochasticGradientDescent(numEpochs, batchSize, learningRate, 0);
		network.train(dataSet, sgd, costFunction);
		//network.stochasticGradientDescent(dataSet, costFunction, numEpochs, batchSize, learningRate);
	}

}
