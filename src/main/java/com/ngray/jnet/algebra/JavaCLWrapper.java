package com.ngray.jnet.algebra;

import org.bridj.Pointer;
import static org.bridj.Pointer.*;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLPlatform.DeviceFeature;

/**
 * Basic JavaCL acceleration for matrix multiplication
 * @author nigelgray
 *
 */
public class JavaCLWrapper {
	
	private static CLContext context = JavaCL.createBestContext(DeviceFeature.GPU);
	private static CLQueue queue = context.createDefaultQueue();	
	
	/**
	 * The matrixMultiply kernel. Both outer loops are executed in parallel, enabling each element of the resultant matrix to
	 * be calculated potentially simultaneously.
	 * There is no bounds-checking - it is up to clients to ensure that their matrices are correctly and consistently sized.
	 */
	private static String matrixMultiplySource = 
	"__kernel void matrixMultiply(__global const float* a, int aRows, int aCols,  __global const float* b, int bRows, int bCols, __global float* res)" +
	"{" +
	"	int i = get_global_id(0);" +
	"	int j = get_global_id(1);" +
	"	res[i * bCols + j] = 0;" +
	"	for (int k = 0; k < aCols; ++k) {" +
	"		res[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];" + 
	"	}" +
	"}";
	
	private static CLKernel kernel = context.createProgram(matrixMultiplySource).createKernel("matrixMultiply");
	
	/**
	 * Multiply the two matrices using JavaCL. Will be much faster than Java for large matrices.
	 * @param lhs
	 * @param rhs
	 * @return
	 */
	public static Matrix multiply(Matrix lhs, Matrix rhs) {
		
		Pointer<Float> lhsPtr = pointerToFloats(lhs.asFloatArray());
		Pointer<Float> rhsPtr = pointerToFloats(rhs.asFloatArray());
		Pointer<Float> outPtr = matrixMultiply(lhsPtr, lhs.getNumRows(), lhs.getNumCols(), rhsPtr, rhs.getNumRows(), rhs.getNumCols());
		Matrix result = Matrix.fromFloatArray(outPtr.getFloats(), lhs.getNumRows(), rhs.getNumCols());
		lhsPtr.release();
		rhsPtr.release();
		outPtr.release();
		return result;
	}
	
	/**
	 * Multiply the vector v by the Matrix M using JavaCL
	 * @param M
	 * @param v
	 * @return
	 */
	public static Vector multiply(Matrix M, Vector v) {
		Pointer<Float> lhsPtr = pointerToFloats(M.asFloatArray());
		Pointer<Float> rhsPtr = pointerToFloats(v.asFloatArray());
		Pointer<Float> outPtr = matrixMultiply(lhsPtr, M.getNumRows(), M.getNumCols(), rhsPtr, v.getSize(), 1);
		Vector result = Vector.fromFloatArray(outPtr.getFloats(), M.getNumRows());
		lhsPtr.release();
		rhsPtr.release();
		outPtr.release();
		return result;
	}
	
	private static Pointer<Float> matrixMultiply(Pointer<Float> a, int aRows, int aCols, Pointer<Float> b, int bRows, int bCols) {		
		assert (aCols == bRows);
		CLBuffer<Float> aBuf = context.createBuffer(CLMem.Usage.Input, a, false);
		CLBuffer<Float> bBuf = context.createBuffer(CLMem.Usage.Input, b, false);
		CLBuffer<Float> outBuf = context.createBuffer(CLMem.Usage.Output, Float.class, aRows * bCols);
		kernel.setArgs(aBuf, aRows, aCols, bBuf, bRows, bCols, outBuf);
		kernel.enqueueNDRange(queue, new int[]{aRows, bCols});
		queue.finish();
		return outBuf.read(queue);
	}

}
