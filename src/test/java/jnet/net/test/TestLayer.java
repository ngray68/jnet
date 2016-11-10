package jnet.net.test;

import junit.framework.TestCase;
import jnet.net.CostFunction;
import jnet.data.DataInstance;
import jnet.net.Layer;
import jnet.net.Vector;

public class TestLayer extends TestCase {

	public void testGetNumNeurons() {
		//fail("Not yet implemented");
		Layer layer = new Layer(5, null);
		assert (layer.getNumNeurons() == 5);
	}

	public void testFeedForward() {
		//fail("Not yet implemented");
		// The test proves that the hidden layer has no activation before the feedForward
		// (activation == null) and that after feedForward the activation vector has been
		// assigned a value
		Layer inputLayer = new Layer(5,null);
		Layer nextLayer = new Layer(10, inputLayer);
		inputLayer.setActivation(new Vector(new Double[] {1.0, 2.0, 3.0, 4.0, 5.0}));
		assert (nextLayer.getActivation() == null);
		nextLayer.feedForward();
		assert (nextLayer.getActivation() != null);
	}

	public void testBackPropagateDataInstanceCostFunction() {
		//fail("Not yet implemented");
		CostFunction costFunction = (expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		Layer previous = new Layer(3, null);
		previous.setActivation(inputs);
		Layer outputLayer = new Layer(3, previous);
		
		assert (outputLayer.getError() == null);
		outputLayer.feedForward();
		outputLayer.backPropagate(instance, costFunction);
		assert (outputLayer.getError() != null);
		
	}

	public void testBackPropagateLayer() {
		//fail("Not yet implemented");
		CostFunction costFunction = (expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		Layer inputLayer = new Layer(3, null);
		inputLayer.setActivation(inputs);
		Layer hiddenLayer = new Layer(10, inputLayer);
		Layer outputLayer = new Layer(3, hiddenLayer);
		
		assert (hiddenLayer.getError() == null);
		hiddenLayer.feedForward();	
		outputLayer.feedForward();
		outputLayer.backPropagate(instance, costFunction);
		hiddenLayer.backPropagate(outputLayer);
		assert (hiddenLayer.getError() != null);
		
	}

	public void testGetBiasGradient() {
		//fail("Not yet implemented");
		CostFunction costFunction = (expOutput, output) -> { return Vector.add(expOutput, Vector.multiply(-1.0, output)); };
		Vector inputs = new Vector(new Double[] {1.0,2.0,1.0});
		Vector expectedOutput = new Vector(new Double[] {0.0,1.0,0.0});
		DataInstance instance = new DataInstance(inputs, expectedOutput);
		Layer previous = new Layer(3, null);
		previous.setActivation(inputs);
		Layer outputLayer = new Layer(3, previous);
		
		assert (outputLayer.getError() == null);
		outputLayer.feedForward();
		outputLayer.backPropagate(instance, costFunction);
		assert (outputLayer.getBiasGradient() == outputLayer.getError());
	}

	public void testGetWeightGradient() {
		fail("Not yet implemented");
	}

	public void testAddBiasGradient() {
		fail("Not yet implemented");
	}

	public void testAddWeightGradient() {
		fail("Not yet implemented");
	}

	public void testGetActivation() {
		fail("Not yet implemented");
	}

	public void testSetActivation() {
		fail("Not yet implemented");
	}

	public void testGetPrevious() {
		fail("Not yet implemented");
	}

	public void testSetPrevious() {
		fail("Not yet implemented");
	}

	public void testGetError() {
		fail("Not yet implemented");
	}

	public void testAdjustWeights() {
		fail("Not yet implemented");
	}

	public void testAdjustBiases() {
		fail("Not yet implemented");
	}

}
