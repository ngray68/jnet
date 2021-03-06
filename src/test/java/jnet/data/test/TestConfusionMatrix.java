package jnet.data.test;

import static org.hamcrest.CoreMatchers.*;
import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import jnet.data.ConfusionMatrix;
import jnet.net.Vector;

public class TestConfusionMatrix {
	
	ConfusionMatrix confusionMatrix;

	@Before
	public void setUp() throws Exception 
	{
		Vector actualOutput = new Vector(new double[]{0.9, 0.1, 0.1});
		Vector expectedOutput = new Vector(new double[]{1.0, 0.0, 0.0});
		confusionMatrix = new ConfusionMatrix(expectedOutput, actualOutput);
		actualOutput = new Vector(new double[]{0.1, 0.9, 0.1});
		expectedOutput = new Vector(new double[]{1.0, 0.0, 0.0});
		confusionMatrix.addDataInstance(expectedOutput, actualOutput);
	}

	@After
	public void tearDown() throws Exception 
	{
	}


	@Test
	public void testAddDataInstance() 
	{
		assertThat("Expect 1 instance, got more than 1", confusionMatrix.getNumInstances(), is(2));
		Vector actualOutput = new Vector(new double[]{0.1, 0.9, 0.1});
		Vector expectedOutput = new Vector(new double[]{1.0, 0.0, 0.0});
		confusionMatrix.addDataInstance(expectedOutput, actualOutput);
		assertThat("Expect 2 instances", confusionMatrix.getNumInstances(), is(3));	
	}

	@Test
	public void testGetAccuracy() 
	{
		assertThat(confusionMatrix.getAccuracy(), is(1.0/2.2));
	}

	@Test
	public void testGetPrecision() 
	{
		Vector precision = confusionMatrix.getPrecision();
		assertThat(precision.getElement(0), is(1.0/(2.2)));
		assertThat(precision.getElement(1), is(0.0));
		assertThat(precision.getElement(2), is(0.0));
	}

	@Test
	public void testGetRecall() 
	{
		Vector recall = confusionMatrix.getRecall();	
		assertThat(recall.getElement(0), is(1.0));
		assertThat(recall.getElement(1), is(0.0));
		assertThat(recall.getElement(2), is(0.0));
	}
}
