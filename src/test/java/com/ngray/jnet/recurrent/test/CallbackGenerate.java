package com.ngray.jnet.recurrent.test;

import com.ngray.jnet.recurrent.Callback;
import com.ngray.jnet.recurrent.DictionaryException;
import com.ngray.jnet.recurrent.RecurrentNeuralNetwork;
import com.ngray.jnet.recurrent.Sequence;
import com.ngray.jnet.recurrent.SequenceException;

public class CallbackGenerate implements Callback {

	private Sequence start;
	private int generatedSequenceLength;
	
	public CallbackGenerate(Sequence start, int generatedSequenceLength) {
		this.start = start;
		this.generatedSequenceLength = generatedSequenceLength;
	}
	@Override
	public void call(RecurrentNeuralNetwork rnn) {
			try {
				Sequence output = rnn.generate(start, generatedSequenceLength);
				String s = Alphabet.convert(Alphabet.createSequenceGenerator().convertSequence(rnn.predict(output)));
				System.out.println(s);
			} catch (SequenceException | DictionaryException e) {
				e.printStackTrace();
			}
	}

}
