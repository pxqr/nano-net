using System;

using Nanon.Math.Activator;
using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer.Convolutional
{
	public class VectorSubsampler : ISingleLayer<Vector, Vector>
	{
		IActivator activator;
		
		Vector downsampled;
		Vector signals;
		Vector outputs;
		
		double weight;
		double gradient;
		
		double bias;
		double biasGradient;
		
		Vector predError;
			
		public VectorSubsampler (int outputSize, IActivator activatorA)
		{
			activator = activatorA;
			
			var range = OptimalInitEpsilon(outputSize, outputSize);
			
			downsampled = new Vector(outputSize);
			signals   = new Vector(outputSize);
			outputs   = new Vector(outputSize);
			
			weight    = Vector.RandomUniform(1, 0.1)[0];
			bias      = Vector.RandomUniform(1, 0.1)[0];
			
			gradient  = 0.0d;
			biasGradient = 0.0d;
			
			predError = new Vector(outputSize * 2);
		}
		
		static double OptimalInitEpsilon(int inputSize, int outputSize)
		{
			return System.Math.Sqrt(6) / System.Math.Sqrt(inputSize + outputSize);
		}

		#region ISingleLayer[Vector,Vector] implementation
		
		public Vector FeedForward (Vector input)
		{
			input.DownsampleBy2(downsampled);
			downsampled.Multiply(weight, signals);
			signals.Add(bias, signals);
			
			Vector.Transform(activator.Activate, signals, outputs);
			return outputs;
		}

		public Vector PropagateBackward (Vector input, Vector predSignal, Vector error)
		{
			var err = weight * error;
			var fan = predError.ZeroCopy();
			err.UpsampleBy2(fan);
			
			Vector.Transform(activator.Derivative, predSignal, predError);
			predError.Multiply(fan, predError);
			return predError;
		}

		public void Gradient (Vector input, Vector outputError)
		{
			gradient     += outputError * downsampled;
			biasGradient += outputError.Sum;
		}

		public void Correct (double coeff)
		{
			//weight += coeff * gradient;
			gradient = 0;
			
			bias   -= coeff * biasGradient;
			biasGradient = 0;
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
	}
}

