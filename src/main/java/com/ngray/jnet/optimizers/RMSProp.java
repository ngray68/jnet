package com.ngray.jnet.optimizers;

import com.ngray.jnet.algebra.Matrix;

public final class RMSProp {

	private Matrix meanSquare;
	private final double decayRate;
	private final double learningRate;
	private final double minimumLearningRateMultiple;
	private final double maximumLearningRateMultiple;
	private final double epsilon;
	
	public RMSProp(int rowDimension, int colDimension, double decayRate, double learningRate, double minimumLearningRateMultiple, double maximumLearningRateMultiple) {
		this.meanSquare = new Matrix(rowDimension, colDimension, 0.0);
		this.decayRate = decayRate;
		this.epsilon = 1.0E-6;
		this.learningRate = learningRate;
		this.minimumLearningRateMultiple = minimumLearningRateMultiple;
		this.maximumLearningRateMultiple = maximumLearningRateMultiple;
	}
	
	
	public void updateMeanSquare(Matrix gradient) throws RMSPropException {
		checkGradient(gradient);
		Matrix gradientSquared = gradient.multiplyElementWise(gradient);
		meanSquare = meanSquare.multiply(decayRate).add(gradientSquared.multiply(1.0-decayRate));
	}
	
	public Matrix getGradientMultiplier() {
		//return getRMS().multiply(learningRate);
		return getRMS();
	}
	
	
	
	private Matrix getRMS() {
		Matrix rms = new Matrix(meanSquare.getNumRows(), meanSquare.getNumCols());
		for (int i = 0; i < meanSquare.getNumRows(); ++i) {
			for (int j = 0; j < meanSquare.getNumCols(); ++j) {
				double val = learningRate/(Math.sqrt(meanSquare.getElement(i, j)) + epsilon);
				val = val < minimumLearningRateMultiple ? minimumLearningRateMultiple : val;
				val = val > maximumLearningRateMultiple ? maximumLearningRateMultiple : val;
				rms.setElement(i, j, val);
			}
		}	
		return rms;
	}
	
	public void printDiagnostics() {
		System.out.println("RMSProp: mean square values");
		System.out.println(meanSquare.toString());
		System.out.println("RMSProp: gradient multiplier");
		System.out.println(getGradientMultiplier().toString());
	}
	
	
	private void checkGradient(Matrix gradient) throws RMSPropException {
		if (gradient == null) {
			throw new RMSPropException("Null gradient matrix passed to RMSProp");
		}
		checkGradientDimension(gradient);
	}
	
	private void checkGradientDimension(Matrix gradient) throws RMSPropException {
		if (gradient.getNumRows() != meanSquare.getNumRows() ||
			gradient.getNumCols() != meanSquare.getNumCols()) {
			throw new RMSPropException("Attempted update of RMSProp optimizer with a gradient of the wrong dimension");
		}
	}
	
	
}
