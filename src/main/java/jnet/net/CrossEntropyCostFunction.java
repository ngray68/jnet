package jnet.net;

public class CrossEntropyCostFunction implements CostFunction {

	@Override
	public double cost(Vector output, Vector expectedOutput) {
		// TODO Auto-generated method stub
		// ylna + (1-y)ln(1-a)
		Vector unit = new Vector(output.getSize(), 1.0);
		double exp1 = Vector.dotProduct(expectedOutput, ln(output));
		double exp2 = Vector.dotProduct(
							Vector.add(unit, Vector.multiply(-1.0, expectedOutput)), 
							ln(Vector.add(unit, Vector.multiply(-1.0, output)))
							);
		return -(exp1 + exp2);
	}

	@Override
	public Vector costPrime(Vector output, Vector expectedOutput) {
		Vector unit = new Vector(output.getSize(), 1.0);
		Vector exp1 = Vector.schurProduct(expectedOutput, reciprocal(output));
		Vector exp2 = Vector.schurProduct(
							Vector.add(unit, Vector.multiply(-1.0, expectedOutput)), 
							reciprocal(Vector.add(unit, Vector.multiply(-1.0, output)))
							);
		return Vector.multiply(-1.0, Vector.add(exp1, Vector.multiply(-1.0, exp2)));
	}
	
	private Vector ln(Vector v) 
	{
		Vector r = new Vector(v.getSize());
		for (int i = 0; i < v.getSize(); ++i) {
			r.setElement(i, Math.log(v.getElement(i)));
		}
		return r;
	}
	
	private Vector reciprocal(Vector v) {
		Vector r = new Vector(v.getSize());
		for (int i = 0; i < v.getSize(); ++i) {
			if (v.getElement(i) != 0) {
				r.setElement(i, 1.0/v.getElement(i));
			} else {
				r.setElement(i, v.getElement(i));
			}
		}
		return r;
	}

}
