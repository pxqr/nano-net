using System;

using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer
{
	public class ConvolutionalLayer : ISingleLayer<Matrix, Matrix>
	{
		
		
		public ConvolutionalLayer ()
		{
			
		}

		#region ISingleLayer[Matrix,Matrix] implementation
		
		public Matrix FeedForward (Matrix input)
		{
			throw new NotImplementedException ();
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

		public void Correct (Vector gradients)
		{
			throw new NotImplementedException ();
		}

		public Vector ZeroGradients ()
		{
			throw new NotImplementedException ();
		}

		public Matrix Signal {
			get {
				throw new NotImplementedException ();
			}
		}

		public Matrix Output {
			get {
				throw new NotImplementedException ();
			}
		}

		#endregion
	}
}

