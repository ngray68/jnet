package com.ngray.jnet.recurrent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.ngray.jnet.algebra.Matrix;
import com.ngray.jnet.algebra.Vector;
import com.ngray.jnet.optimizers.RMSProp;
import com.ngray.jnet.optimizers.RMSPropException;


/**
 * A single layer RecurrentNeuralNetwork
 * @author nigelgray
 *
 */
public final class RecurrentNeuralNetwork {
	
	/**
	 * Matrices U, V, W are the weight matrices of the RNN
	 * Matrix U is of dimension n x m, where:
	 * n == no of neurons in the network
	 * m == size of the input vectors for which the network has been constructed.
	 * Matrix W is of dimension n x n
	 * Matrix V is of size m x n (ie. opposite to U and W) 
	 * Each vector in (hidden) state list is of size n (ie. one value per neuron)
	 */
	private Matrix U;
	private Matrix V;
	private Matrix W;
	private List<Vector> state;
	
	private Matrix dLossdU;
	private Matrix dLossdV;
	private Matrix dLossdW;
	
	private RMSProp rmsPropU;
	private RMSProp rmsPropV;
	private RMSProp rmsPropW;
	
	private final int networkDimension;
	private final int inputDimension;
	private final int outputDimension;
	private final double softmaxTemp;
	private double rmsDecay;
	private double minStepRate;
	private double maxStepRate;
	
	public static class Builder {
		
		private int inputDimension;
		private int outputDimension;
		private int networkDimension;
		private double softmaxTemp;
		private double rmsDecay;
		private double minStepRate;
		private double maxStepRate;
		
		public Builder() {
			softmaxTemp = 1.0;
			rmsDecay = 0.9;
			minStepRate = 0.1;
			maxStepRate = 1.0;
		}
		public RecurrentNeuralNetwork build() throws RecurrentNeuralNetworkException {
			if (inputDimension == 0 || outputDimension == 0 || networkDimension == 0) {
				throw new RecurrentNeuralNetworkException("Please specify non-zero dimension before building the network");
			}
			return new RecurrentNeuralNetwork(networkDimension,inputDimension,outputDimension,softmaxTemp,rmsDecay,minStepRate, maxStepRate);
		}
		
		public Builder setInputDimension(int inputDimension) {
			this.inputDimension = inputDimension;
			return this;
		}
		
		public Builder setOutputDimension(int outputDimension) {
			this.outputDimension = outputDimension;
			return this;
		}
		
		public Builder setNetworkDimension(int networkDimension) {
			this.networkDimension = networkDimension;
			return this;
		}
		
		public Builder setSoftmaxTemp(double softmaxTemp) {
			this.softmaxTemp = softmaxTemp;
			return this;
		}
		
		public Builder setRMSDecay(double rmsDecay) {
			this.rmsDecay = rmsDecay;
			return this;
		}
		
		public Builder setMinStepRate(double minStepRate) {
			this.minStepRate = minStepRate;
			return this;
		}
		
		public Builder setMaxStepRate(double maxStepRate) {
			this.maxStepRate = maxStepRate;
			return this;
		}
	};
	// Constructors
	public RecurrentNeuralNetwork(int networkDimension, int inputDimension, int outputDimension) {
		this(networkDimension, inputDimension, outputDimension, 1.0, 1.0, 0.1, 1.0);
	}
	
	public RecurrentNeuralNetwork(int networkDimension, int inputDimension, int outputDimension, double softmaxTemp, double rmsDecay, double minStepRate, double maxStepRate) {
		U = new Matrix(networkDimension, inputDimension, new Random(), 1.0/Math.sqrt(inputDimension));
		W = new Matrix(networkDimension, networkDimension, new Random(), 1.0/Math.sqrt(networkDimension));
		V = new Matrix(outputDimension, networkDimension, new Random(), 1.0/Math.sqrt(networkDimension));
	
		state = new ArrayList<>();
		this.networkDimension = networkDimension;
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		this.softmaxTemp = softmaxTemp;
		this.rmsDecay = rmsDecay;
		this.minStepRate = minStepRate;
		this.maxStepRate = maxStepRate;
		initializeGradients();	
	}
	
	// Public interface ///////////////////////////////////////////////////////////
	/**
	 * This method evaluates the inputSequence
	 * and returns the outputSequence. The RNNs state is reset
	 * at the end of the process
	 * @param inputSequence
	 * @return
	 * @throws SequenceException
	 */
	public Sequence evaluate(Sequence inputSequence) throws SequenceException{
		Sequence output = feedForward(inputSequence);
		state.clear();
		return output;
	}

	/**
	 * Calculate the loss of the actualOutput sequence vs the expectedOutput sequence
	 * Sequences must be of the same length, and contain vectors of the same dimension
	 * @param actualOutput
	 * @param expectedOutput
	 * @return
	 */
	public double calculateLoss(Sequence actualOutput, Sequence expectedOutput) {
		int N = actualOutput.getLength();
		
		// empty sequences always have zero loss
		if (N == 0) {
			return 0.0;
		}
	
		return calculateTotalLoss(actualOutput, expectedOutput)/N;
	}
	
	public double calculateLoss(List<Sequence> outputs, List<Sequence> expectedOutputs) {
		// use the cross-entropy loss function
		// ie. -expectedOutput * ln(actualOutput)

		if (outputs.size() == 0) {
			return 0.0;
		}
		Iterator<Sequence> itOut = outputs.iterator();
		Iterator<Sequence> itExp = expectedOutputs.iterator();
		double loss = 0.0;
		int N = 0;
		while (itOut.hasNext() && itExp.hasNext()) {
			Sequence out = itOut.next();
			Sequence exp = itExp.next();
			loss += calculateTotalLoss(out, exp);
			N += out.getLength();
		}
		return loss/N;
	}
	
	private double calculateTotalLoss(Sequence actualOutput, Sequence expectedOutput) {
		double loss = 0.0;
		Iterator<Vector> actualIter = actualOutput.getIterator();
		Iterator<Vector> expIter = expectedOutput.getIterator();
		while (actualIter.hasNext() && expIter.hasNext()) {
			loss = loss - expIter.next().dotProduct(ln(actualIter.next()));
		}
		return loss;
	}
	
	/**
	 * Given a sequence s of probability vectors, return a sequence of
	 * one-hot vectors which is the sequence of most likely values
	 * @param s
	 * @return
	 * @throws SequenceException
	 */
	public Sequence predict(Sequence s) throws SequenceException {
		List<Vector> elements = new ArrayList<>();
		for (Iterator<Vector> iter = s.getIterator(); iter.hasNext(); ) {
			elements.add(predict(iter.next()));
		}
		return Sequence.newSequence(elements);
	}
	
	/**
	 * If v is a vector of probabilities return the one-hot vector representing
	 * the max probability
	 * @param v
	 * @return
	 */
	public Vector predict(Vector v) {
		// given a vector of probabilities, collapse all onto the most likely
		double max = 0.0;
		int maxIndex = -1;
		for (int i = 0; i < v.getSize(); ++i) {
			if (v.getElement(i) > max) {
				max = v.getElement(i);
				maxIndex = i;
			}
		}
		Vector result = new Vector(v.getSize(), 0.0);
		result.setElement(maxIndex, 1.0);
		return result;
	}
		
	/**
	 * Generate a sequence starting with start using the trained network
	 * @param start
	 * @param totalSequenceLength
	 * @return
	 * @throws SequenceException 
	 */
	public Sequence generate(Sequence start, int totalSequenceLength) throws SequenceException {
		if (start.getLength() >= totalSequenceLength) {
			throw new SequenceException("Can't generate a new sequence of total length " + totalSequenceLength + 
					" from starting sequence that is longer");
		}
		List<Vector> output = new ArrayList<>();
		Sequence result = Sequence.copySequence(start);
		
		int i = 0;
		Vector next = null;
		for (Iterator<Vector> it = start.getIterator(); it.hasNext(); ) {	
			next = it.next();
			if (it.hasNext()) {
				calculateState(next);
			}
			++i;
		}
		
		for (int j = i; j < totalSequenceLength; ++j) {
			Vector thisOutput = softmax(V.multiply(calculateState(next)));
			output.add(thisOutput);
			next = predict(thisOutput);
		}
		
		state.clear();
		return result.join(Sequence.newSequence(output));
	}
	
	/**
	 * Train the network
	 * @param dataSet
	 * @param epochs
	 * @param batchSize
	 * @param learningRate
	 * @param maxBackPropSteps
	 * @throws RMSPropException 
	 */
	public void train(DataSet dataSet, int epochs, int batchSize, double learningRate, int maxBackPropSteps, double lossThreshold, Callback...callbacks) throws RMSPropException {
		
		rmsPropU = new RMSProp(networkDimension, inputDimension, rmsDecay, learningRate, minStepRate, maxStepRate);
		rmsPropW = new RMSProp(networkDimension, networkDimension, rmsDecay, learningRate, minStepRate, maxStepRate);
		rmsPropV = new RMSProp(outputDimension, networkDimension, rmsDecay, learningRate, minStepRate, maxStepRate);
		
		List<Sequence> inputs = dataSet.getInputData();	
		Collections.shuffle(inputs);
		for (int i = 0; i < epochs; ++i) {
			
			for (Callback callback : callbacks) {
				callback.call(this);
			}
			System.out.println("Epoch " +i);

			Collections.shuffle(inputs);
			double currentAverageLoss = 0.0;
			int numBatches = inputs.size()/batchSize;
			
			for (int j = 0; j < numBatches; ++j) {
				System.out.println("Epoch " + i +" Batch " + j);
				//List<Sequence> batch = inputs.subList(j*batchSize, (j+1)*batchSize);
				List<Sequence> batch = inputs.subList(j*batchSize, j*batchSize + 5);
				List<Sequence> expectedOutputs = new ArrayList<>();
				List<Sequence> outputs = new ArrayList<>();
				int scaleFactor = 0;
				for (Iterator<Sequence> it = batch.iterator(); it.hasNext(); ) {
					Sequence input = it.next();
					scaleFactor += input.getLength();
					try {
						Sequence expectedOutput = dataSet.getExpectedOutput(input);
						Sequence output = feedForward(input);
						backPropagateThroughTime(input, output, expectedOutput, maxBackPropSteps);	
						outputs.add(output);
						expectedOutputs.add(expectedOutput);
						state.clear();
					} catch (SequenceException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				
				// calculate the loss for this batch
				double loss = calculateLoss(outputs, expectedOutputs);
				currentAverageLoss = (currentAverageLoss*j + loss)/(j+1);
				System.out.println("Learning rate: " + learningRate + "\tLoss: " + loss + "\tRunning Avg: " + currentAverageLoss);
				// primitive early-stopping
				if(loss < lossThreshold) {
					System.out.println("Stopping training");
					initializeGradients();
					state.clear();	
					return;
				}
				
				System.out.println("Adjusting weights");
				adjustWeights(learningRate, scaleFactor);
				initializeGradients();
				state.clear();		
			}
		}
		System.out.println("Training complete");
	}

	// Private helper functions ////////////////////////////////////////////////////////////////////////
	private void adjustWeights(double learningRate, double sequenceLength) throws RMSPropException {
		
		rmsPropU.updateMeanSquare(dLossdU.multiply(1.0/sequenceLength));
		rmsPropV.updateMeanSquare(dLossdV.multiply(1.0/sequenceLength));
		rmsPropW.updateMeanSquare(dLossdW.multiply(1.0/sequenceLength));
		
		U = U.subtract(dLossdU.multiply(1.0/sequenceLength).multiplyElementWise(rmsPropU.getGradientMultiplier()));
		V = V.subtract(dLossdV.multiply(1.0/sequenceLength).multiplyElementWise(rmsPropV.getGradientMultiplier()));
		W = W.subtract(dLossdW.multiply(1.0/sequenceLength).multiplyElementWise(rmsPropW.getGradientMultiplier()));
	}
	
	/**
	 * This method calculates and returns the output sequence
	 * and also calculates the hidden state
	 * when the given inputSequence is supplied to the RNN
	 * In practice the method will never throw a SequenceException
	 * since if the inputSequence is valid, the output sequence is guaranteed
	 * to be valid - and we can never construct an invalid sequence
	 * @param inputSequence
	 * @return
	 * @throws SequenceException 
	 */
	private Sequence feedForward(Sequence inputSequence) throws SequenceException {
		if (inputSequence == null) {
			throw new SequenceException("Can't evaluate null sequence");
		}
		
		List<Vector> output = new ArrayList<>();	
		Iterator<Vector> inputIter = inputSequence.getIterator();
		while (inputIter.hasNext()) {
			output.add(softmax(V.multiply(calculateState(inputIter.next()))));
		}
		return Sequence.newSequence(output);
	}
	
	private void backPropagateThroughTime(Sequence input, Sequence actualOutput, Sequence expectedOutput, double maxBackPropSteps) {
	
		int T = input.getLength();
		for (int t = T-1; t >= 0; --t) {
			Vector thisActual = actualOutput.get(t);
			Vector thisExpected = expectedOutput.get(t);
			Vector thisState = state.get(t);
			Vector outputDelta = thisActual.subtract(thisExpected);
			dLossdV = dLossdV.add(outputDelta.dyadicProduct(thisState));
			
			Vector delta_t = V.multiply(jacobian_s_f(thisState)).transpose().multiply(outputDelta);
			for (int t2 = t; t2 > Math.max(0, t-maxBackPropSteps); --t2) {
				Vector nextState = state.get(t2-1);
				dLossdW = dLossdW.add(delta_t.dyadicProduct(nextState));
				dLossdU = dLossdU.add(delta_t.dyadicProduct(input.get(t2)));
				delta_t = W.transpose().multiply(jacobian_s_f(nextState)).multiply(delta_t);
			}
		}
		//dLossdV = dLossdV.multiply(1.0/T);
		//dLossdW = dLossdW.multiply(1.0/T);
		//dLossdU = dLossdU.multiply(1.0/T);
	}
		
	private Vector calculateState(Vector input) {
		// calculate the next state vector as a function of
		// the current input and the last state
		// s(t) = tanh(Ux(t) + Ws(t-1))
		Vector nextState = null;
		if (state.size() == 0) {
			nextState = tanh(U.multiply(input));
			state.add(nextState);
		} else {
			Vector prevState = state.get(state.size() - 1);
			nextState = tanh(U.multiply(input).add(W.multiply(prevState)));
			state.add(nextState);		
		}
		return nextState;
	}
	
	private Vector tanh(Vector x) {
		Vector result = new Vector(x.getSize());
		for (int i = 0; i < x.getSize(); ++i) {
			result.setElement(i, Math.tanh(x.getElement(i)));
		}
		return result;
	}
	
	private Vector softmax(Vector x) {
		Vector unit = new Vector(x.getSize(), 1.0);
		return exp(x.multiply(1.0/softmaxTemp)).multiply(1.0/exp(x.multiply(1.0/softmaxTemp)).dotProduct(unit));
	}
	
	private Vector exp(Vector x) {
		Vector result = new Vector(x.getSize());
		for (int i = 0; i < x.getSize(); ++i) {
			result.setElement(i, Math.exp(x.getElement(i)));
		}
		return result;
	}
	
	private Vector ln(Vector x) {
		Vector result = new Vector(x.getSize());
		for (int i = 0; i < x.getSize(); ++i) {
			result.setElement(i, Math.log(x.getElement(i)));
		}
		return result;
	}
	
	private Matrix jacobian_s_f(Vector s) {
		// s = tanh(f)
		// the jacobian ds/df is I-s^2
		Matrix J = new Matrix(s.getSize(), s.getSize());
		for (int i = 0; i < s.getSize(); ++i) {
			J.setElement(i, i, 1 - s.getElement(i)*s.getElement(i));
		}
		return J;
	}
	
	private void initializeGradients() {
		dLossdU = new Matrix(networkDimension, inputDimension);
		dLossdW = new Matrix(networkDimension, networkDimension);
		dLossdV = new Matrix(outputDimension, networkDimension);	
	}
		
	private void performGradientCheck(Sequence input, Sequence expectedOutput) throws SequenceException {
		
		double h = 0.0001;
		double tolerance = 0.01;
		
		// check dLossdU
		System.out.println("Performing gradient check");
		System.out.println("Checking dLossdU...");
		
		for (int i = 0; i < U.getNumRows(); ++i) {
			for (int j = 0; j < U.getNumCols(); ++j) {
	
				double param = U.getElement(i, j);
				U.setElement(i, j, param - h);
				double lossminus = calculateLoss(evaluate(input), expectedOutput);
				U.setElement(i, j, param + h);
				double lossplus = calculateLoss(evaluate(input), expectedOutput);
				double estimatedGradient = (lossplus - lossminus)/(2*h);
				
				double error = Math.abs(dLossdU.getElement(i, j) - estimatedGradient);
				System.out.print("dLossdU(" + i + "," + j + "): ");
				System.out.print("Actual grad: " + dLossdU.getElement(i, j));
				System.out.print(" Estimated grad: " + estimatedGradient);
				System.out.print(" Error: " + error);
				
				if (error <= tolerance) {
					System.out.println(" PASSED");
				} else {
					System.out.println(" FAILED");
				}
				
				U.setElement(i, j, param);
			}
		}
		/*
		
		// check dLossdW
		System.out.println("Checking dLossdW...");
		for (int i = 0; i < W.getNumRows(); ++i) {
			for (int j = 0; j < W.getNumCols(); ++j) {
	
				double param = W.getElement(i, j);
				W.setElement(i, j, param - h);
				double lossminus = calculateLoss(evaluate(input), expectedOutput);
				W.setElement(i, j, param + h);
				double lossplus = calculateLoss(evaluate(input), expectedOutput);
				double estimatedGradient = (lossplus - lossminus)/(2*h);
				
				double error = Math.abs(dLossdW.getElement(i, j) - estimatedGradient);
				System.out.print("dLossdW(" + i + "," + j + "): ");
				System.out.print("Actual grad: " + dLossdW.getElement(i, j));
				System.out.print("\tEstimated grad: " + estimatedGradient);
				System.out.print("\tError: " + error);
				
				if (error <= tolerance) {
					System.out.println(" PASSED");
				} else {
					System.out.println(" FAILED");
				}
				
				W.setElement(i, j, param);
			}
		}
		
		// check dLossdV
		/*
		System.out.println("Checking dLossdV...");
		for (int i = 0; i < V.getNumRows(); ++i) {
			for (int j = 0; j < V.getNumCols(); ++j) {
	
				double param = V.getElement(i, j);
				V.setElement(i, j, param - h);
				double lossminus = calculateLoss(evaluate(input), expectedOutput);
				V.setElement(i, j, param + h);
				double lossplus = calculateLoss(evaluate(input), expectedOutput);
				double estimatedGradient = (lossplus - lossminus)/(2*h);
				
				double error = Math.abs(dLossdV.getElement(i, j) - estimatedGradient);
				System.out.print("dLossdV(" + i + "," + j + "): ");
				System.out.print("Actual grad: " + dLossdV.getElement(i, j));
				System.out.print("\tEstimated grad: " + estimatedGradient);
				System.out.print("\tError: " + error);
				
				if (error <= tolerance) {
					System.out.println(" PASSED");
				} else {
					System.out.println(" FAILED");
				}
				
				V.setElement(i, j, param);
			}
		}*/
		
	}
		
}