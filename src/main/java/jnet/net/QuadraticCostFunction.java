package jnet.net;

import com.ngray.jnet.algebra.Vector;

public class QuadraticCostFunction implements CostFunction {

	public QuadraticCostFunction() 
	{	
	}
	
	@Override
	public double cost(Vector output, Vector expectedOutput) 
	{
		//Vector diff = Vector.add(expectedOutput, Vector.multiply(-1.0, output));
		Vector diff = expectedOutput.subtract(output);
		return 0.5 * diff.dotProduct(diff);
	}

	@Override
	public Vector costPrime(Vector output, Vector expectedOutput) 
	{
		return output.subtract(expectedOutput);
		//return Vector.add(output, Vector.multiply(-1.0, expectedOutput));
	}

}
