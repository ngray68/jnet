package jnet.net.test;

import junit.framework.TestCase;
import jnet.net.CostFunction;
import jnet.data.DataInstance;
import jnet.data.DataSet;
import jnet.net.Network;
import jnet.net.Vector;

public class TestNetwork extends TestCase {

	public void testStochasticGradientDescent() {
		//fail("Not yet implemented");
		Network network = new Network(new int[]{3,10,3});
		
		
		CostFunction costFunction = (expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		DataSet dataSet = new DataSet();
		int batchSize = 5;
		double learningRate = 0.005;
		int numEpochs = 1;
		
		
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		
		for (int i = 0; i < 10; ++i) {
			dataSet.addInstance(new DataInstance(inputs, expectedOutput));
		}
		
		network.stochasticGradientDescent(dataSet, costFunction, numEpochs, batchSize, learningRate);
		
		// TODO add evaluate method on DataInstance
		// TODO add evaluate method on DataSet
	}

	public void testFeedForward() {
		//fail("Not yet implemented");
		Network network = new Network(new int[]{3,10,3});
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		network.feedForward(instance);
	
	}

	public void testBackPropagate() {
		//fail("Not yet implemented");
		CostFunction costFunction = (expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		Network network = new Network(new int[]{3,10,3});
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		network.feedForward(instance);
		network.backPropagate(instance, costFunction);
	}

	public void testCalculateGradient() {
		//fail("Not yet implemented");
		CostFunction costFunction = (expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		Network network = new Network(new int[]{3,10,3});
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		network.feedForward(instance);
		network.backPropagate(instance, costFunction);
		network.calculateGradient();
	}

}
