package edu.stanford.rsl.tutorial.dmip;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.numerics.SimpleVector.VectorNormType;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.IJ;
import ij.ImageJ;
import ij.plugin.filter.Convolver;
import ij.process.FloatProcessor;


/**
 * Introduction to the CONRAD Framework
 * Exercise 1 of Diagnostic Medical Image Processing (DMIP)
 * @author Marco Boegel
 *
 */

public class Intro {
	
	
	public static void gridIntro(){
				
		//Define the image size
		int imageSizeX = 256;
		int imageSizeY = 256;
	
		//Define an image
		//Hint: Import the package edu.stanford.rsl.conrad.data.numeric.Grid2D
		//TODO
	
		//Draw a circle
		int radius = 50;
		//Set all pixels within the circle to 100
		int insideVal = 100;
	
		//TODO
		//TODO
		//TODO
		
		//Show ImageJ GUI
		ImageJ ij = new ImageJ();
		//Display image
		//TODO
		
		//Copy an image
		//TODO
		//copy.show("Copy of circle");
		
		
		//Load an image from file
		String filename = "D:/02_lectures/DMIP/exercises/2014/matlab_intro/mr12.dcm";
		//TODO. Hint: Use IJ and ImageUtil
		//mrImage.show();
		
		//convolution
		//TODO
		//TODO
		
		//define the kernel. Try simple averaging 3x3 filter
		int kw = 3;
		int kh = 3;
		float[] kernel = new float[kw*kh];
		for(int i = 0; i < kernel.length; i++)
		{	
			kernel[i] = 1.f / (kw*kh);
		}
		
		//TODO
			
		
		//write an image to disk, check the supported output formats
		String outFilename ="D:/02_lectures/DMIP/exercises/2014/matlab_intro/mr12out.tif";
		//TODO
	}
	
	
	public static void signalIntro()
	{
		//How can I plot a sine function sin(2*PI*x)?
		double stepSize = 0.01;
		int plotLength = 500;
		
		double[] y = new double[plotLength];
		
		for(int i = 0; i < y.length; i++)
		{
			y[i]=Math.sin(2.0 * Math.PI*stepSize*(double)i);
			
		}
		
		VisualizationUtil.createPlot(y).show();
		double[] x = new double [plotLength];
		for(int i = 0; i < x.length; i++)
		{
			x[i] = (double) i * stepSize;
		}
		
		VisualizationUtil.createPlot(x, y, "sin(x)", "x", "y").show();		
		
	}
	
	public static void basicIntro()
	{
		//Display text
		System.out.println("Creating a vector: v1 = [1.0; 2.0; 3.0]");
		
		//create column vector
		//TODO
		//System.out.println("v1 = " + v1.toString());
		
		//create a randomly initialized vector
		SimpleVector vRand = new SimpleVector(3);
		//TODO
		//System.out.println("vRand = " + vRand.toString());
		
		//create matrix M 3x3  1 2 3; 4 5 6; 7 8 9
		SimpleMatrix M = new SimpleMatrix();
		//TODO
		//System.out.println("M = " + M.toString());
		
		//determinant of M
		//System.out.println("Determinant of matrix m: " + TODO );
		
		//transpose M
		//TODO
		//copy matrix
		//TODO
		//transpose M inplace
		//TODO
		
		//get size
		int numRows = 0;
		int numCols = 0;
		//TODO
		
		//access elements of M
		System.out.println("M: ");
		for(int i = 0 ; i < numRows; i++)
		{
			for(int j = 0; j < numCols; j++)
			{
				//TODO
				//System.out.print(element + " ");
			}
			System.out.println();
		}
		
		//Create 3x3 Matrix of 1's
		SimpleMatrix Mones = new SimpleMatrix(3,3);
		//TODO
		//Create a 3x3 Matrix of 0's
		SimpleMatrix Mzeros = new SimpleMatrix(3,3);
		//TODO
		//Create a 3x3 Identity matrix
		SimpleMatrix Midentity = new SimpleMatrix(3,3);
		//TODO
		
		//Matrix multiplication
		//TODO
		//System.out.println("M^T * M = " + ResMat.toString());
		

		//Matrix vector multiplication
		//TODO
		//System.out.println("M * v1 = " + resVec.toString());
		
		
		//Extract the last column vector from matrix M
		//SimpleVector colVector = M.getCol(2);
		//Extract the 1x2 subvector from the last column of matrix M
		//TODO
		//System.out.println("[m(0)(2); m(1)(2)] = " + subVector);
		
		//Matrix elementwise multiplication
		//TODO
		//System.out.println("M squared Elements: " + MsquaredElem.toString());
		
		//round vectors
		SimpleVector vRandCopy = new SimpleVector(vRand);
		System.out.println("vRand         = " + vRandCopy.toString());
		
		vRandCopy.floor();
		System.out.println("vRand.floor() = " + vRandCopy.toString());
		
		vRand.ceil();
		System.out.println("vRand.ceil()  = " + vRand.toString());
		
		//min, max, mean
		//double minV1 = v1.min();
		//double maxV1 = v1.max();
		//System.out.println("Min(v1) = " + minV1 + " Max(v1) = " + maxV1);
		
		//for matrices: iterate over row or column vectors
		SimpleVector maxVec = new SimpleVector(M.getCols());
		for(int i = 0; i < M.getCols(); i++)
		{
			maxVec.setElementValue(i, M.getCol(i).max());
		}
		double maxM = maxVec.max();
		System.out.println("Max(M) = " + maxM);
		
		
		
		//Norms
		//TODO matrix L1
		//TODO vector L2
		//System.out.println("||M||_F = " + matrixNormL1);
		//System.out.println("||colVec||_2 = " + vecNormL2);
		
		//get normalized vector
		//TODO
		//normalize vector in-place
		//TODO
		//System.out.println("Normalized colVector: " + colVector.toString());
		
		
		//SVD
		SimpleMatrix A = new SimpleMatrix(3,3);package edu.stanford.rsl.tutorial.dmip;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

/**
 * 
 * Exercise 2 of Diagnostic Medical Image Processing (DMIP)
 * @author Marco Boegel
 *
 */
public class SVDandFT {
	
	public static void invertSVD(SimpleMatrix A)
	{
		
					
		System.out.println("A = " + A.toString());

		//Compute the inverse of A without using inverse()				
		//TODO 
		DecompositionSVD svd = new DecompositionSVD(A);
		

		//Check output: re-compute A = U * S * V^T
		SimpleMatrix temp = SimpleOperators.multiplyMatrixProd(svd.getU(), svd.getS());
		SimpleMatrix A2 = SimpleOperators.multiplyMatrixProd(temp, svd.getV().transposed());
		System.out.println("U * S * V^T: " + A2.toString());
		
		//Moore-Penrose Pseudoinverse defined as V * Sinv * U^T
		//Compute the inverse of the singular matrix S
		SimpleMatrix Sinv = new SimpleMatrix( svd.getS().getRows(), svd.getS().getCols());
		Sinv.zeros();

		int size = Math.min(Sinv.getCols(), Sinv.getRows());
		SimpleVector SinvDiag = new SimpleVector( size);
		
		for(int i = 0; i < Sinv.getRows(); i++ ) {
			double val = 1.0 / svd.getS(). getElement(i, 1);
			Sinv.setElementValue(i, i, val);
			
		}
		
		Sinv.setDiagValue(SinvDiag);
		
		//Compare our implementation to svd.getreciprocalS()
		System.out.println("Sinv = " + Sinv.toString());
		System.out.println("Srec = " +svd.getreciprocalS().toString());
		
		
		SimpleMatrix tempInv = SimpleOperators.multiplyMatrixProd(svd.getV(), svd.getreciprocalS());
		SimpleMatrix Ainv = SimpleOperators.multiplyMatrixProd(tempInv, svd.getU().transposed());
		System.out.println("Ainv        = " + Ainv.toString());
		System.out.println("A.inverse() = " + A.inverse(InversionType.INVERT_SVD));
		
		//Condition number
		//TODO
		double cond = svd.cond();
		System.out.println("Cond(A) = " + cond);
		
		//introduce a rank deficiency
		double eps = 10e-3;
		SimpleMatrix Slowrank = new SimpleMatrix(svd.getS());
		int sInd = Math.min(svd.getS().getRows(), svd.getS().getCols());
		for(int i = 0; i < sInd; i++)
		{
			double val = svd.getS().getElement(i, i);
			if (val > eps) {
				Slowrank.setElementValue(i, i, val);			
			}			
		}
		
		SimpleMatrix templowrank = SimpleOperators.multiplyMatrixProd(svd.getU(), Slowrank);
		SimpleMatrix Alowrank = SimpleOperators.multiplyMatrixProd(templowrank, svd.getV().transposed());
		System.out.println("A rank deficient = " + Alowrank.toString());
		
		
		
		//Show that a change in a vector b of only 0.1% can lead to 240% change in the result of A*x=b
		//Consider A as already defined
		//And vector b
		SimpleVector b = new SimpleVector(1.001, 0.999, 1.001);
		
		//solve for x
		SimpleVector x = SimpleOperators.multiply(Ainv, b);
		System.out.println(x.toString());
		
		//if we set vector b to the vector consisting of 1's, we imply a 0.1% change
		SimpleVector br = new SimpleVector(1,1,1);
		
		
		SimpleVector xr = SimpleOperators.multiply(Ainv, br);
		System.out.println(xr.toString());
		//We want only the difference caused by the change	
		SimpleVector xn = SimpleOperators.subtract(xr, x);
		
		//Alternatively:
		//SimpleVector bn = SimpleOperators.subtract(br, b);
		//SimpleVector xn = SimpleOperators.multiply(Ainv, bn);
				
		System.out.println("Modification of x " + SimpleOperators.divideElementWise(xn, x).toString());
	}
	
	public static void optimizationProblem1(SimpleMatrix A, int rankDeficiency)
	{
		System.out.println("Optimization Problem 1");
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// Optimization problem I
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// Let us consider the following problem that appears in many image
		// processing and computer vision problems:
		// We computed a matrix A out of sensor data like an image. By theory
		// the matrix A must have the singular values s1, s2, . . . , sp, where
		// p = min(m, n). Of course, in practice A does not fulfill this constraint.
		// Problem: What is the matrix A0 that is closest to A (according to the
		// Frobenius norm) and has the required singular values?


		// Let us assume that by theoretical arguments the matrix A is required
		// to have a rank deficiency of one, and the two non-zero singular values
		// are identical. The matrix A0 that is closest to A according to the
		// Frobenius norm and fulfills the above requirements is:

		// Build mean of first two singular values to make them identical
		// Set third singular value to zero
		
		DecompositionSVD svd = new DecompositionSVD(A);
		int rank = svd.rank();
		
		int newRank = rank - rankDeficiency;
		
		
		
		//Compute mean of remaining singular values
		double mean = 0;
		
		for(int i = 0; i < newRank; i++) {
			mean += svd.getS().getElement(i,1);
		}
		mean /= newRank;
		
		//Create new Singular matrix
		SimpleMatrix Slowrank = new SimpleMatrix(svd.getS().getRows(), svd.getS().getCols());
		Slowrank.zeros();
		//Fill in remaining singular values with the mean.
		
		for(int i = 0; i < newRank; i++) {
			Slowrank.setElementValue(i, i, mean);
		}
		
		//compute A0
		SimpleMatrix temp = SimpleOperators.multiplyMatrixProd(svd.getU(), Slowrank);
		SimpleMatrix A0 = SimpleOperators.multiplyMatrixProd(temp, svd.getV().transposed());
		
		System.out.println("A0 = " + A0.toString());
		
		double normA = A.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		double normA0 = A0.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		
		System.out.println("||A||_F  = " + normA);
		System.out.println("||A0||_F = " + normA0);
		
		
	}
	
	public static void optimizationProblem4(double[] xCoords, double[] yCoords)
	{
		System.out.println("Optimization Problem 4");
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//Optimization problem IV
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// The Moore-Penrose pseudo-inverse is required to find the
		// solution to the following optimization problem:
		// Compute the regression line through the following 2-D points:
		// We have sample points (x_i,y_i) and want to fit a line model
		// y = mx + t (linear approximation) through the points
		// y = A*b
		// Note: y = m*x +t*1  Thus: A contains (x_i 1) in each row, and y contains the y_i coordinates
		// We need to solve for b = (m t)
		
		SimpleMatrix A = new SimpleMatrix(xCoords.length, 2);
		SimpleVector aCol = new SimpleVector(xCoords.length);
		SimpleVector y = new SimpleVector(xCoords.length);
		for(int i = 0; i < xCoords.length; i++)
		{
			aCol.setElementValue(i, xCoords[i]);
			y.setElementValue(i, yCoords[i]);
		}
		A.setColValue(0, aCol);
		SimpleVector aColOnes = new SimpleVector(xCoords.length);
		aColOnes.ones();
		A.setColValue(1, aColOnes);
		
		
		
		// Note: We need to use SVD matrix inversion because the matrix is not square
		// more samples (7) than unknowns (2) -> overdetermined system -> least-square
		// solution.
		
		//get solution for b
		
		SimpleMatrix Ainv = A.inverse(InversionType.INVERT_SVD);
		SimpleVector b = SimpleOperators.multiply(Ainv, y);
	
		
		LinearFunction lFunc = new LinearFunction();
		lFunc.setM(b.getElement(0));
		lFunc.setT(b.getElement(1));
		VisualizationUtil.createScatterPlot(xCoords, yCoords, lFunc).show();
		
	}
	
	public static void optimizationProblem2(SimpleMatrix b)
	{
		System.out.println("Optimization Problem 2");
		// Estimate the matrix A in R^{2,2} such that for the following vectors
		// the optimization problem gets solved:
		// sum_{i=1}^{4} b'_{i} A b_{i} and ||A||_{F} = 1
		// The objective function is linear in the components of A, thus the whole
		// sum can be rewritten in matrix notation:
		// Ma = 0
		// where the measurement matrix M is built from single elements of the
		// sum.
		// Given the four points we get four equations and therefore four rows in M:
		
		SimpleMatrix M = new SimpleMatrix( b.getCols(), 4);
		
		for(int i = 0; i < b.getCols(); i++)
		{	
			// i-th column of b contains a vector. First entry of M is x^2
			//Setup measurement matrix M
			M.setElementValue(i, 0, Math.pow(b.getElement(0, i), 2.f));
			M.setElementValue(i, 1, b.getElement(0, 1) * b.getElement(1, i));
			M.setElementValue(i, 2, M.getElement(i, 1));
			M.setElementValue(i, 3, Math.pow(b.getElement(01, i), 2.f));
		}
		
		// TASK: estimate the matrix A
		// HINT: Nullspace
		DecompositionSVD svd = new DecompositionSVD(M);
		int lastCol = svd.getV().getCols();
		SimpleVector a = svd.getV().getCol(lastCol);
		
		
		//We need to reshape the vector back to the desired 2x2 matrix form
		SimpleMatrix A = new SimpleMatrix(2,2);
		A.setColValue(0, a.getSubVec(0, 2));
		A.setColValue(1, a.getSubVec(2, 2));
		
		
		//check if Frobenius norm is 1.0
		double normF = A.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		System.out.println("||A||_F = " + normF);
		
		//check solution
		
		SimpleVector temp = SimpleOperators.multiply(M, a);
		double result = SimpleOperators.multiplyInnerProd(a, temp);
		System.out.println("Minimized error: " + result);
	}
	
	public static void optimizationProblem3(Grid2D image, int rank)
	{
		//%
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// Optimization problem III
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// The SVD can be used to compute the image matrix of rank k that best
		// approximates an image. Figure 1 shows an example of an image I
		// and its rank-1-approximation I0 = u_{1} * s_{1} * v'_{1}.
		
		//In order to apply the svd, we first need to transfer our Grid2D image to a matrix
		SimpleMatrix I = new SimpleMatrix(image.getHeight(), image.getWidth());
		
		//NOTE: indices of matrix and image are reversed
		for(int i = 0; i < image.getHeight(); i++)
		{
			for(int j = 0; j < image.getWidth(); j++)
			{
				double val = image.getAtIndex(j, i);
				I.setElementValue(i, j, val);
			}
		}		
		
		DecompositionSVD svd = new DecompositionSVD(I);
		
		//output images
		Grid3D imageRanks = new Grid3D(image.getWidth(), image.getHeight(), rank);
	
		//Create Rank k approximations
		for(int k = 0; k < rank; k++)
		{
			SimpleVector us = svd.getU().getCol(k).multipliedBy(svd.getSingularValues()[k]);
			SimpleMatrix Iapprox = SimpleOperators.multiplyOuterProd(us, svd.getV().getCol(k));
			
			
			
	
		
			//Transfer back to grid
			Grid2D imageRank = new Grid2D(image.getWidth(),image.getHeight());
			for(int i = 0; i < image.getHeight(); i++)
			{
				for(int j = 0; j < image.getWidth(); j++)
				{
					if(k == 0)
					{
						imageRank.setAtIndex(j, i, (float) Iapprox.getElement(i, j));
			
					}
					else
					{
						imageRank.setAtIndex(j, i, imageRanks.getAtIndex(j, i, k-1) + (float) Iapprox.getElement(i, j));
					}
					
					
				}
			}
			imageRanks.setSubGrid(k, imageRank);
		}
		
		imageRanks.show();
		
		
		//Direct estimation of rank K 
		//TODO
		//TODO
		//Transfer back to grid
		Grid2D imageRankK = new Grid2D(image.getWidth(), image.getHeight());
		for(int i = 0; i < image.getHeight(); i++)
		{
			for(int j = 0; j < image.getWidth(); j++)
			{
					//imageRankK.setAtIndex(j, i, (float) Iapprox.getElement(i, j));
			}
		}
		imageRankK.show();

	
	}
	
	public static void fourierExercise(Grid2D image)
	{
		//TODO complex image
		// Important: Grid2DComplex enlarges the original image to the next power of 2
		Grid2DComplex imageC = new Grid2DComplex(image);
		imageC.show();
		
		//Apply 2-D discrete fourier transform
		//Puts the DC component of the signal in the upper left corner of the FFT
		imageC.transformForward();
		imageC.show("Shepp-Logan FFT");
		
		imageC.fftshift();
		imageC.show("Shepp-Logan FFTShift");
		
		//Visualize log transformed FFT log(1+|FFTshift(image)|)
		Grid2D logFFT = new Grid2D(imageC.getMagnSubGrid(0, 0, imageC.getWidth(), imageC.getHeight()));
		logFFT.getGridOperator().addBy(logFFT, 1.f);
		logFFT.getGridOperator().log(logFFT);
		logFFT.show("Shepp-Logan FFTShift log");
		
		// Important: Grid2DComplex enlarges the original image to the next power of 2
		// When transforming back, make sure to prune the image size to your original image
		imageC.transformInverse();
		Grid2D imageTransf = imageC.getMagnSubGrid(0, 0, image.getWidth(), image.getHeight());
		imageTransf.show();
	}

	public static void main(String[] args) {
		
		ImageJ ij = new ImageJ();
		//Create matrix A 
		// 11 10  14
		// 12 11 -13
		// 14 13 -66
		SimpleMatrix A = new SimpleMatrix(3,3);
		A.setRowValue(0, new SimpleVector(11, 10,  14));
		A.setRowValue(1, new SimpleVector(12, 11, -13));
		A.setRowValue(2, new SimpleVector(14, 13, -66));
		
		invertSVD(A);
		optimizationProblem1(A, 1);
		
		//Data for problem 4 from lecture slides
		double[] xCoords = new double[]{3.f, 2.f, 1.f, 0.f, -1.f, -1.f, -2.f};
		double[] yCoords = new double[]{2.f, 1.f, 2.f, 0.f, 1.f, -1.f, -1.f};
	
		optimizationProblem4(xCoords, yCoords);
		
		
		
		// Data for problem 2 from lecture slides
		SimpleMatrix vectors = new SimpleMatrix(2,4);
		vectors.setColValue(0, new SimpleVector(1.f, 1.f));
		vectors.setColValue(1, new SimpleVector(-1.f, 2.f));
		vectors.setColValue(2, new SimpleVector(1.f, -3.f));
		vectors.setColValue(3, new SimpleVector(-1.f, -4.f));
		
		//hier kommt ein fehler
		//optimizationProblem2(vectors);
		
		//Load an image from file
		String filename = "/proj/i5dmip/zi79moji/Reconstruction/CONRAD/src/edu/stanford/rsl/tutorial/dmip/yu_fill.jpg";
		Grid2D image = ImageUtil.wrapImagePlus(IJ.openImage(filename)).getSubGrid(0);
		image.show();
		
		int rank = 100;		
		optimizationProblem3(image, rank);
		
		
		//Load Data for Fourier Transform exercise
		//1. Start the ReconstructionPipelineFrame (src/apps/gui/)
		//2. In the Pipeline window, go to edit configuration and press "center volume" and save
		//3. In the ImageJ window, navigate to Plugins->CONRAD->Create Numerical Phantom
		//4. Choose Metric Volume Phantom
		//5. Choose Shepp Logan Phantom
		//6. Save the resulting volume. In the ImageJ window, File-Save As->Tiff...
		
		String filenameShepp = "/proj/i5dmip/zi79moji/Reconstruction/CONRAD/src/Shepp-Logan Phantom.tif";
		Grid3D sheppLoganVolume = ImageUtil.wrapImagePlus(IJ.openImage(filenameShepp));
		//To work with a 2-D image, select slice 160
		Grid2D sheppLoganImage = sheppLoganVolume.getSubGrid(160);
		sheppLoganImage.show();
		fourierExercise(sheppLoganImage);
		
		

	}

}

		A.setRowValue(0, new SimpleVector(11, 10,  14));
		A.setRowValue(1, new SimpleVector(12, 11, -13));
		A.setRowValue(2, new SimpleVector(14, 13, -66));
				
		System.out.println("A = " + A.toString());
		
		//TODO SVD
		
		//print singular matrix
		//System.out.println(svd.getS().toString());
		
		//get condition number
		//System.out.println("Condition number of A: " + TODO );
		
		//Re-compute A = U * S * V^T
		//SimpleMatrix temp = SimpleOperators.multiplyMatrixProd(svd.getU(), svd.getS());
		//SimpleMatrix A2 = SimpleOperators.multiplyMatrixProd(temp, svd.getV().transposed());
		//System.out.println("U * S * V^T: " + A2.toString());
		
	}

	public static void main(String arg[])
	{
		basicIntro();
		gridIntro();
		signalIntro();
	}
}
