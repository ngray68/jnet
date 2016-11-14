package jnet.algorithm;

import jnet.data.DataSet;
import jnet.net.CostFunction;
import jnet.net.Network;

public interface LearningAlgorithm {
	
	public void execute(Network network, DataSet trainingSet, DataSet validationSet, CostFunction costFunction);
}
