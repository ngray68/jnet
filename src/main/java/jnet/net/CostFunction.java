package jnet.net;

public interface CostFunction {

	public Vector costPrime(Vector output, Vector expectedOutput);
	
}
