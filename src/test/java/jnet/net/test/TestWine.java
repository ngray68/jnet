package jnet.net.test;

import junit.framework.TestCase;
import jnet.net.CostFunction;
import jnet.data.DataSet;
import jnet.net.Network;
import jnet.net.Vector;

public class TestWine extends TestCase {

	private String dataFileName = "./src/test/resources/wine.data";
	private DataSet dataSet;
	
	protected void setUp() throws Exception {
		super.setUp();
		dataSet = new DataSet();
		dataSet.readFromFile(dataFileName);
		dataSet.normalize();
	}

	protected void tearDown() throws Exception {
		super.tearDown();
	}
	
	public void testWine() {
		//fail("Not yet implemented");
		Network network = new Network(new int[] {13, 17, 3});
		CostFunction costFunction = (output, expOutput) -> { return Vector.add(output, Vector.multiply(-1.0, expOutput)); };
		int numEpochs = 5000;
		double learningRate = 0.1;
		int batchSize = 20;
		network.stochasticGradientDescent(dataSet, costFunction, numEpochs, batchSize, learningRate);
	}

}
