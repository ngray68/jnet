package jnet.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import jnet.net.Vector;

public class DataSetLoader {
	
	public static enum FileFormat {
		CSV,
		UNSUPPORTED
	};
	
	/**
	 * Call this method to load a DataSet object from a file.
	 * Will throw if the file does not exist at the specified location, the file is in
	 * an unsupported format, or if there is an error during the file read operation
	 * @param filename
	 * @param fileFormat
	 * @param lineFormat
	 * @return the DataSet object
	 * @throws DataException 
	 */
	public static DataSet loadFromFile(String filename, String fileFormat, String lineFormat) throws DataException
	{
		if (checkFileFormat(fileFormat) == FileFormat.UNSUPPORTED) {
			throw new DataException("File format" + fileFormat + " unsupported");
		}
		DataSet dataSet = DataSet.create();
		try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
			String nextLine = null;
			while ((nextLine = reader.readLine()) != null) {
				dataSet.addInstance(parseCsvLine(nextLine, lineFormat));
			}
		} catch (IOException e) {
			throw new DataException(e);
		}
		return dataSet;
	}
	
	/**
	 * Call this method to load a DataSet object from a file.
	 * Will throw if the file does not exist at the specified location, the file is in
	 * an unsupported format, or if there is an error during the file read operation
	 * @param filename
	 * @param fileFormat
	 * @param numExpectedOutputs
	 * @return the DataSet object
	 * @throws DataException 
	 */
	public static DataSet loadFromFile(String filename, String fileFormat, int numExpectedOutputs) throws DataException
	{
		return loadFromFile(filename, fileFormat, numExpectedOutputs, 0.8, 0.1);
		/*
		if (checkFileFormat(fileFormat) == FileFormat.UNSUPPORTED) {
			throw new DataException("File format" + fileFormat + " unsupported");
		}
		DataSet dataSet = DataSet.create();
		try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
			String nextLine = null;
			while ((nextLine = reader.readLine()) != null) {
				dataSet.addInstance(parseCsvLine(nextLine, numExpectedOutputs));
			}
		} catch (IOException e) {
			throw new DataException(e);
		}
		return dataSet;*/
	}
	
	public static DataSet loadFromFile(String filename, String fileFormat, int numExpectedOutputs, double trainingFraction, double validationFraction) throws DataException
	{
		if (checkFileFormat(fileFormat) == FileFormat.UNSUPPORTED) {
			throw new DataException("File format" + fileFormat + " unsupported");
		}
		DataSet dataSet = DataSet.create(trainingFraction, validationFraction);
		try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
			String nextLine = null;
			while ((nextLine = reader.readLine()) != null) {
				dataSet.addInstance(parseCsvLine(nextLine, numExpectedOutputs));
			}
		} catch (IOException e) {
			throw new DataException(e);
		}
		return dataSet;
	}

	/**
	 * Call this method to determine the file format
	 * @param fileFormat
	 * @return enum value for the file format
	 */
	public static FileFormat checkFileFormat(String fileFormat) 
	{
		if (!fileFormat.equals("csv"))
				return FileFormat.UNSUPPORTED;
		return FileFormat.CSV;
	}
	
	/**
	 * Called by loadFromFile to parse a line of csv to a DataInstance object
	 * Throws if the lineFormat string doesn't match the comma-separated line
	 * @param line
	 * @param lineFormat
	 * @return DataInstance object
	 * @throws DataException
	 */
	private static DataInstance parseCsvLine(String line, String lineFormat) throws DataException 
	{
		String[] tokens = line.split(",");
		char[] format = lineFormat.toCharArray();
		if (format.length != tokens.length) {
			throw new DataException("Supplied line format string is inconsistent with file contents");
		}
		
		List<Double> inputList = new ArrayList<>();
		List<Double> outputList = new ArrayList<>();
		for (int i = 0; i < tokens.length; ++i) {
			Double value = Double.parseDouble(tokens[i]);
			switch (format[i]) {
			case 'I':
				inputList.add(value);
				break;
			case 'E':
				outputList.add(value);
				break;
			default:
				throw new DataException("Illegal character" + format[i] + " in line format string");
			}
		}
		
		Vector outputs = new Vector(outputList);
		Vector inputs = new Vector(inputList);
		return new DataInstance(inputs, outputs);
		
	}
	
	private static DataInstance parseCsvLine(String line, int numExpectedOutputValues)
	{
		Double[] array = new Double[numExpectedOutputValues];
		List<Double> outputList = new ArrayList<>(Arrays.asList(array));
		Collections.fill(outputList, 0.0);
		
		String[] values = line.split(",");
		outputList.set((int)Double.parseDouble(values[0]), 1.0);
	
		// Note - will this preserve the order? Quite important!
		List<String> inputStrings = new ArrayList<String>(Arrays.asList(values).subList(1, values.length));
		List<Double> inputList= inputStrings.stream().map(Double::parseDouble).collect(Collectors.toList());
		
		Vector outputs = new Vector(outputList);
		Vector inputs = new Vector(inputList);
		return new DataInstance(inputs, outputs);		
	}
	
	
}
