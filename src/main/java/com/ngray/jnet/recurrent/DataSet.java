package com.ngray.jnet.recurrent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public final class DataSet {
	
	private final List<Sequence> inputSequences;
	private final Map<Sequence, Sequence> expectedOutputs;
	
	public DataSet(List<Sequence> inputs, Map<Sequence, Sequence> expectedOutputs) {
		this.inputSequences = Collections.unmodifiableList(inputs);
		this.expectedOutputs = Collections.unmodifiableMap(expectedOutputs);
	}
	
	public List<Sequence> getInputData() {
		return new ArrayList<>(inputSequences);
	}
	
	public Sequence getExpectedOutput(Sequence input) throws SequenceException {
		if (expectedOutputs.containsKey(input)) {
			return expectedOutputs.get(input);
		}
		
		throw new SequenceException("The specified input has no expected output defined in the data set");
	}
	
	
}
