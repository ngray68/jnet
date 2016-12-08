package jnet.net;

public class QuadraticCostFunction implements CostFunction {

	public QuadraticCostFunction() 
	{	
	}
	
	@Override
	public double cost(Vector output, Vector expectedOutput) 
	{
		
		Vector diff = Vector.add(expectedOutput, Vector.multiply(-1.0, output));
		return 0.5 * Vector.dotProduct(diff, diff);
	}

	@Override
	public Vector costPrime(Vector output, Vector expectedOutput) 
	{
		return Vector.add(output, Vector.multiply(-1.0, expectedOutput));
	}

}
