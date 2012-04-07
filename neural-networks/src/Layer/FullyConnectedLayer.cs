using System.Collections.Generic;

using Nanon.Math.Linear;
using Nanon.Math.Activator;
	
namespace Nanon.NeuralNetworks.Layer
{
	public class FullyConnectedLayer : ISingleLayer<Vector, Vector>
	{
		// state
		IActivator activator;
		Matrix weights;
		
		// "cache"
		Vector signals;
		Vector outputs;  
		Matrix gradient;
		
		public static IActivator defaultActivator = new Nanon.Math.Activator.Tanh();
		
		public FullyConnectedLayer(int inputSize, int outputSize, IActivator activationFunction)
		{
			// "(+ 1)" because of bias term
			var eps =  OptimalInitEpsilon(inputSize + 1, outputSize);
			weights = Matrix.RandomUnform(inputSize + 1, outputSize, eps);
			
			outputs = new Vector(outputSize);
			signals = new Vector(outputSize);
			
			activator = activationFunction;
			
			gradient = new Matrix(inputSize + 1, outputSize);
		}		
		
		static double OptimalInitEpsilon(int inputSize, int outputSize)
		{
			return System.Math.Sqrt(6) / System.Math.Sqrt(inputSize + outputSize);
		}
		
		#region ILayer[Vector,Vector] implementation
		
		//					prop eqs
		//  y = f (W * x + intercept)
		//  preconditions : input length = InputSize
		//
		public Vector FeedForward(Vector input)
		{
			// signals = weights * (1 `concat` input);
		    Matrix.MultiplyVerticalWithBias(weights, input, signals);
			// output = f (signals);
			Vector.Transform(activator.Activate, signals, outputs);
			
			return outputs;
		}
		
		//  				backprop eqs
		//  errorPred = (transpose W * error) * f' signal
		//  gradientW = error * transpose yPred
		//
		//  preconditions : error lenght = OutputSize
		//
		public Vector PropagateBackward(Vector predSignal, Vector predOutput, Vector error)
		{
			// der = f'(predSignal)
			var der = predSignal.ZipWith(predOutput, activator.Derivative);
			var predError = predOutput.ZeroCopy();
			
			// predError = W' * error			
			Matrix.MultiplyHorizontalWithoutBias(error, weights, predError);
			
			// predError = (W' * error) * f'(predSignal)
			predError.Multiply(der, predError);
			
			return predError;
		}
		
		public Vector Signal 
		{ 
			get 
			{
				return signals;	
			}
		}
		
		public Vector Output
		{
			get 
			{
				return outputs;
			}
		}
		
		public void FindGradient(Vector input, Vector outputError)
		{
			//Matrix.MultiplyWithTrasposedWithBias(outputError, input, gradient);
			Matrix.MultiplyWithTrasposed(outputError, Vector.Prepend(input), gradient);
		}
		
		public Vector Gradient()
		{
			return gradient.ToVector;
		}
		
		public void Correct(Vector grads)
		{    
			weights.SubInplace(grads);
		}
		
		public Vector ZeroGradients()
		{ 
			return weights.ZeroCopy().ToVector;	
		}
		
		public double this [int i] 
		{ 
			get
			{
				return weights[i];	
			}
			set
			{
				weights.Cells[i] = value;
			}
		}
		
		public int WeightsSize  
		{ 
			get
			{
				return weights.Size;
			}
		} 
		
		#endregion
	}
}