/**
 * 
 */
package jnet.test.data;

import static org.junit.Assert.*;
import org.junit.Test;

import jnet.data.DataException;
import jnet.data.DataInstance;
import jnet.data.DataSet;
import jnet.data.DataSetLoader;

/**
 * @author nigelgray
 *
 */
public class TestDataSetLoader {

	private String fileName = "./src/test/resources/wine.csv";
	private String fileFormat = "csv";
	private String lineFormat = "EEEIIIIIIIIIIIII";
	private String nonexistent = "somenonexistentfilename.data";
	private String corruptedFileName = "./src/test/resources/corrupted_wine.data";
	

	/**
	 * Test method for {@link jnet.data.DataSetLoader#loadFromFile(java.lang.String, java.lang.String, java.lang.String)}.
	 */
	@Test
	public void testLoadFromFile() 
	{
		try {
			DataSet dataSet  = DataSetLoader.loadFromFile(fileName, fileFormat, lineFormat);
			assert (dataSet.getNumInstances() > 0);
			DataInstance instance = dataSet.getIterator().next();
			assertNotNull("Failure - detected null instance in DataSet", instance);
			assertNotNull("Failure - detected null inout vector in DataInstance", instance.getInputs());
			assertNotNull("Failure - detected null output vector in DataInstance", instance.getExpectedOutputs());
		} catch (DataException e) {
			assertFalse("Exception thrown from DataSetLoader.loadFromFile", true);
		}
	}
	
	/**
	 * Test method for {@link jnet.data.DataSetLoader#loadFromFile(java.lang.String, java.lang.String, java.lang.String)}.
	 * when called with a null filename
	 */
	@Test(expected = NullPointerException.class)
	public void testLoadFromFileNull() throws DataException {
		DataSetLoader.loadFromFile(null, fileFormat, lineFormat);
	}
	
	/**
	 * Test method for {@link jnet.data.DataSetLoader#loadFromFile(java.lang.String, java.lang.String, java.lang.String)}.
	 * when called with a non-existent filename
	 * @throws DataException 
	 */
	@Test(expected = DataException.class)
	public void testLoadFromFileNonExistent() throws DataException {
		DataSetLoader.loadFromFile(nonexistent, fileFormat, lineFormat);
	}

	/**
	 * Test method for {@link jnet.data.DataSetLoader#loadFromFile(java.lang.String, java.lang.String, java.lang.String)}.
	 * when called with a file whose format is not supported
	 * @throws DataException 
	 */
	@Test(expected = DataException.class)
	public void testLoadFromFileUnrecognisedFormat() throws DataException {
		DataSetLoader.loadFromFile(fileName, "blah", lineFormat);
	}
	
	/**
	 * Test method for {@link jnet.data.DataSetLoader#loadFromFile(java.lang.String, java.lang.String, java.lang.String)}.
	 * when called with a file whose data is corrupted or inconsistent
	 * @throws DataException 
	 */
	@Test(expected = DataException.class)
	public void testLoadFromFileCorrupted() throws DataException {
		DataSetLoader.loadFromFile(corruptedFileName , fileFormat, lineFormat);
	}
	
	
	/**
	 * Test method for {@link jnet.data.DataSetLoader#checkFileFormat(java.lang.String)}.
	 */
	@Test
	public void testCheckFileFormat() {
		assert(DataSetLoader.checkFileFormat(fileFormat) == DataSetLoader.FileFormat.CSV);
		assert(DataSetLoader.checkFileFormat("blah") == DataSetLoader.FileFormat.UNSUPPORTED);
	}

}
