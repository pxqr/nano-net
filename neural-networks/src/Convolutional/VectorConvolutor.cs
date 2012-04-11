using System;

using Nanon.Math.Activator;
using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer.Convolutional
{
	public class VectorConvolutor : ISingleLayer<Vector, Vector>
	{
		IActivator activator;
		
		Vector signals;
		Vector outputs;
		
		double bias;
		double biasGradient;
		Vector weights;
		Vector gradients;

		Vector predError;
		
		public VectorConvolutor (int inputSize, int outputSize, IActivator activatorA)
		{
			var kernelSize = inputSize - outputSize + 1;
			
			activator = activatorA;
			
			signals = new Vector(outputSize);
			outputs = new Vector(outputSize);
			
			bias         = 0.0d;
			biasGradient = 0.0d;
			
			var eps = OptimalInitEpsilon(inputSize + 1, outputSize);
			weights      = Vector.RandomUniform(kernelSize, eps);
			gradients    = new Vector(kernelSize);
			
			predError = new Vector(inputSize);
		}
		
		static double OptimalInitEpsilon(int inputSize, int outputSize)
		{
			return System.Math.Sqrt(6) / System.Math.Sqrt(inputSize + outputSize);
		}
		
		#region ISingleLayer[Vector,Vector] implementation
		
		public Vector FeedForward (Vector input)
		{
			input.Convolve(weights, signals);
			signals.Add(bias, signals);
			Vector.Transform(activator.Activate, signals, outputs);
			return outputs;
		}

		public Vector PropagateBackward (Vector input, Vector predSignal, Vector error)
		{
			throw new NotImplementedException ();
		}

		public void Gradient (Vector inputs, Vector outputError)
		{
			inputs.Involve(outputError, gradients);
			biasGradient += outputError.Sum;
		}

		public void Correct (double coeff)
		{
			gradients.Multiply(coeff, gradients);
			weights.Sub(gradients, weights);
			gradients.SetToZero();
			bias -= coeff * biasGradient;
			biasGradient = 0.0d;
		}

		public Vector Signal {
			get {
				return signals;
			}
		}

		public Vector Output {
			get {
				return outputs;
			}
		}
		
		#endregion

		public Vector Weights {
			get {
				return this.weights;
			}
		}

		public Vector Gradients {
			get {
				return this.gradients;
			}
		}
	}
}