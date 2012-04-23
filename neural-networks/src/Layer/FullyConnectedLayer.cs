using System.Collections.Generic;

using Nanon.Math.Linear;
using Nanon.Math.Activator;
	
namespace Nanon.NeuralNetworks.Layer
{
	public class FullyConnectedLayer : SingleLayer<Vector, Vector>
	{
		// state
		Matrix weights;
		
		// "cache"
		Matrix gradients;
		Vector predError;
		
		// balancer
		double inputFactor;
		
		public static IActivator defaultActivator = new Nanon.Math.Activator.Tanh();
		
		public FullyConnectedLayer(int inputSize, int outputSize, IActivator activationFunction) : 
			base(activationFunction) 
		{
			// "(+ 1)" because of bias term
			var eps = OptimalInitEpsilon(inputSize + 1, outputSize);
			weights = Matrix.RandomUnform(inputSize + 1, outputSize, eps);
			
			signals = new Vector(outputSize);
			outputs = new Vector(outputSize);
			predError = new Vector(inputSize);
			
			//inputFactor = 1.0d * System.Math.Sqrt(inputSize + 1);
			gradients = new Matrix(inputSize + 1, outputSize);
		}		
		
		#region ILayer[Vector,Vector] implementation
		
		//					prop eqs
		//  y = f (W * x + intercept)
		//  preconditions : input length = InputSize
		//
		public override Vector FeedForward(Vector input)
		{
		    Matrix.MultiplyVerticalWithBias(weights, input, signals);
			signals.Transform(activator.Activate, outputs);
			
			return outputs;
		}
		
		//  				backprop eqs
		//  errorPred = (transpose W * error) * f' signal
		//  gradientW = error * transpose yPred
		//
		//  preconditions : error lenght = OutputSize
		//
		public override Vector PropagateBackward(Vector input, Vector error)
		{	
			// der = f'(input)
			var der = input.Map(activator.Derivative);
			
			// predError = W' * error			
			Matrix.MultiplyHorizontalWithoutBias(error, weights, predError);
			
			// predError = (W' * error) * f'(predSignal)
			predError.Mul(der, predError);
			
			return predError;
		}
		
		public override void Gradient(Vector input, Vector outputError)
		{
			Matrix.MultiplyWithTrasposedWithBiasAdd(outputError, input, gradients);
		}
		
		public override void Correct(double coeff)
		{    
			gradients.Mul(coeff * inputFactor, gradients);
		    weights.Sub(gradients, weights);
			gradients.SetToZero();
		}
		
		#endregion
	}
}