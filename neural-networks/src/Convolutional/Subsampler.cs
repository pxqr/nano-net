using System;

using Nanon.Math.Linear;
using Nanon.Math.Activator;

namespace Nanon.NeuralNetworks.Layer.Convolutional
{
	public class Subsampler<T> : SingleLayer<T, T> where T : IMatrix<T>
	{
		protected T downsampled;
		protected T predError;
		
		double weight       = -0.5d;
		double gradient     = 0.0d;
		
		double bias         = 0.15d;
		double biasGradient = 0.0d;
		
		protected double weightFactor = 1.0d;
		protected double biasFactor   = 1.0d;
		
		internal Subsampler(IActivator activator): base(activator) 	{}
		
		public static Subsampler<Vector> VectorSubsampler(int inputSize, int outputSize, IActivator activatorA)
		{
			if (inputSize != 2 * outputSize)
				throw new ArgumentException("Output size should be half of input size.");
			
			var layer = new Subsampler<Vector>(activatorA);
			
			layer.downsampled = new Vector(outputSize);
			layer.signals     = new Vector(outputSize);
			layer.outputs     = new Vector(outputSize);
			layer.predError   = new Vector(outputSize * 2);
			
			return layer;
		}

		#region implemented abstract members of Nanon.NeuralNetworks.Layer.SingleLayer[T,T]
		
		public override T FeedForward (T input)
		{
			input.DownsampleBy2(downsampled);
			downsampled.Mul(weight, signals);
			signals.Add(bias, signals);
			
			signals.Transform(activator.Activate, outputs);
			return outputs;
		}

		public override T PropagateBackward (T input, T error)
		{
			var tmp   = error.Copy();
			error.Mul(weight, tmp);
			var upscaled = predError.ZeroCopy();
			tmp.UpsampleBy2(upscaled);
			
			input.Transform(activator.Derivative, predError);
			predError.Mul(upscaled, predError);
			return predError;
		}

		public override void Gradient (T input, T outputError)
		{
			gradient     += (outputError.Unwind * downsampled.Unwind) * weightFactor;
			biasGradient += (outputError.Sum * biasFactor);
		}

		public override void Correct (double coeff)
		{
			weight -= coeff * gradient;
			gradient = 0;
			
			bias   -= coeff * biasGradient;
			biasGradient = 0;
		}
		
		#endregion
	}
	
	public class MatrixSubsampler : Subsampler<Matrix>
	{
		public MatrixSubsampler(int inputWidth, int inputHeight, int outputWidth, int outputHeight, IActivator activator) : 
			base(activator)
		{
			if (inputWidth != outputWidth * 2 || inputHeight != outputHeight * 2)
				throw new ArgumentException("Incorrect sizes!");
			
			predError = new Matrix(inputWidth, inputHeight);
			
			downsampled = new Matrix(outputWidth, outputHeight);
			signals = new Matrix(outputWidth, outputHeight);
			outputs = new Matrix(outputWidth, outputHeight);
			
			biasFactor = 1 / (double)signals.Size;
			weightFactor = biasFactor;
		}
	}
}

