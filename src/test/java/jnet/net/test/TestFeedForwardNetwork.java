package jnet.net.test;

import junit.framework.TestCase;
import jnet.net.CostFunction;
import jnet.net.QuadraticCostFunction;
import jnet.net.NetworkException;
import jnet.algorithm.StochasticGradientDescent;
import jnet.data.DataInstance;
import jnet.data.DataSet;
import jnet.net.FeedForwardNetwork;
import jnet.net.Vector;

public class TestFeedForwardNetwork extends TestCase {

	public void testTrain() {
		//fail("Not yet implemented");
		FeedForwardNetwork network = new FeedForwardNetwork(new int[]{3,10,3});
		
		
		CostFunction costFunction = new QuadraticCostFunction(); //(expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		DataSet dataSet = DataSet.create();	
		
		Vector inputs = new Vector(new double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new double[] {0.0,1.0,0.0});
		
		for (int i = 0; i < 10; ++i) {
			dataSet.addInstance(new DataInstance(inputs, expectedOutput));
		}
		
		StochasticGradientDescent sgd = new StochasticGradientDescent(1,5, 0.005, 0);
		try {
			network.train(dataSet.getTrainingSubset(), dataSet.getValidationSubset(), sgd, costFunction);
		} catch (NetworkException e) {
			assertTrue("Test failed", false);
			e.printStackTrace();
		}
	}
	
	public void testValidateOrTest() {
		fail("Not yet implemented");
		
	}

	public void testEvaluate() {
		//fail("Not yet implemented");
		FeedForwardNetwork network = new FeedForwardNetwork(new int[]{3,10,3});
		Vector inputs = new Vector(new double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		try {
			network.evaluate(instance);
		} catch (NetworkException e) {
			assertTrue("Test failed", false);
			e.printStackTrace();
		}
	
	}

	/* Move to tests for StochasticGradientDescent
	public void testBackPropagate() {
		//fail("Not yet implemented");
		CostFunction costFunction = (expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		MultilayerPerceptronNetwork network = new MultilayerPerceptronNetwork(new int[]{3,10,3});
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		network.feedForward(instance);
		network.backPropagate(instance, costFunction);
	}

	public void testCalculateGradient() {
		//fail("Not yet implemented");
		CostFunction costFunction = (expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		MultilayerPerceptronNetwork network = new MultilayerPerceptronNetwork(new int[]{3,10,3});
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		network.feedForward(instance);
		network.backPropagate(instance, costFunction);
		network.calculateGradient();
	}*/

}
