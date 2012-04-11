using System;

using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer.Convolutional
{
	public class MatrixConvolutor : ISingleLayer<Matrix, Matrix>
	{
		public MatrixConvolutor ()
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

		public void Gradient (Matrix input, Matrix outputError)
		{
			throw new NotImplementedException ();
		}

		public void Correct (double gradients)
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

