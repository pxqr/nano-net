using System;

using Nanon.Math.Activator;
using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer
{
	public class CLayer : ISingleLayer<Matrix, Matrix>
	{
		IActivator activator;
		Matrix weights; // kernel
		
		// "cache"
		Matrix signals;
		Matrix outputs;  
		Matrix gradient;
		
		public CLayer(int inputWidth, int inputHeight, 
		              int outputWidth, int outputHeight)
		{
			var kernelWidth  = inputWidth  - outputWidth  + 1;
			var kernelHeight = inputHeight - outputHeight + 1;
			
			var eps =  OptimalInitEpsilon(kernelWidth, kernelHeight);
			weights = Matrix.RandomUnform(kernelWidth, kernelHeight, eps);
			
			outputs = new Matrix(outputWidth, outputHeight);
			signals = new Matrix(outputWidth, outputHeight);
			
			activator = new Tanh();
			
			gradient = new Matrix(kernelWidth, kernelHeight);
		}
		
		static double OptimalInitEpsilon(int inputSize, int outputSize)
		{
			return System.Math.Sqrt(6) / System.Math.Sqrt(inputSize + outputSize);
		}
		
		#region ISingleLayer[Matrix,Matrix] implementation
		
		public Matrix FeedForward (Matrix input)
		{
			Matrix.Convolve(input, weights, signals);
			outputs.Transform(activator.Activate);
			return outputs;
		}

		public Matrix PropagateBackward (Matrix input, Matrix predSignal, Matrix error)
		{
			throw new NotImplementedException ();
		}

		public void FindGradient (Matrix input, Matrix outputError)
		{
			throw new NotImplementedException ();
		}

		public Vector Gradient ()
		{
			throw new NotImplementedException ();
		}

		public void Correct(Vector gradients)
		{
			var grads = new Matrix(weights.Width, 
			                       weights.Height, 
			                       gradient.Cells);
			weights += grads;
		}

		public Nanon.Math.Linear.Vector ZeroGradients ()
		{
			return gradient.ZeroCopy().ToVector;
		}

		public Matrix Signal {
			get {
				return signals;
			}
		}

		public Matrix Output {
			get {
				return outputs;
			}
		}
		#endregion
	}
}

