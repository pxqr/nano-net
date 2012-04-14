using System;

using Nanon.Math.Linear;
using Nanon.Math.Activator;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public class OutputLayer<T> : ICompositeLayer<T, T> where T : IMatrix<T>
	{
		public OutputLayer(){	}

		#region ICompositeLayer[InputT,OutputT] implementation
			
		public T FeedForward(T input)
		{
			return input;
		}
		
		public void Backprop(T input, T error) {	}
		
		public T PropagateBackward(T input, T error)
		{
			//error.Transform(new Tanh().Derivative, error);
			return error;
		}

		public void Gradient(T a, T b) {	}
		public void Correct(double coeff) {		}
		
		#endregion
	}
}