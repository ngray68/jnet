package com.ngray.jnet.recurrent.test;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;

import com.ngray.jnet.algebra.Matrix;
import com.ngray.jnet.optimizers.RMSProp;
import com.ngray.jnet.optimizers.RMSPropException;

public final class TestRMSProp {

	@Test
	public void test() throws RMSPropException {
		// Not really a unit test as such
		// Needs visual inspection of output to check validity as written
		 RMSProp rmsProp = new RMSProp(2, 2, 0.9, 0.25, 0.2, 2.0);
		 
		 for (int i = 0; i < 5; ++i) {
			 Matrix gradient = new Matrix(2, 2, new Random());
			 System.out.println("Gradient: ");
			 System.out.println(gradient.toString());
			 rmsProp.updateMeanSquare(gradient);
			 rmsProp.getGradientMultiplier();
			 rmsProp.printDiagnostics();
			 System.out.println("");
		 }
	}

}
