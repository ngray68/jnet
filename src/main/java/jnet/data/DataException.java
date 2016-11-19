package jnet.data;

import java.io.IOException;

public class DataException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DataException(String string) {
		super(string);
	}

	public DataException(IOException e) {
		super(e);
	}

}
