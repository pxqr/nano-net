using System;

using Nanon.Math.Activator;
using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer
{
	public class Contractor : ISingleLayer<Matrix[], Vector>
	{
		IActivator activator;
		ISingleLayer<Matrix, Matrix>[] layers;
		
		public Contractor ()
		{
			
		}

		#region ISingleLayer[Matrix[],Vector] implementation
		
		public Vector FeedForward (Matrix[] input)
		{
			
		}

		public Matrix[] PropagateBackward (Matrix[] input, Matrix[] predSignal, Vector error)
		{
			throw new NotImplementedException ();
		}

		public void FindGradient (Matrix[] input, Vector outputError)
		{
			throw new NotImplementedException ();
		}

		public Vector Signal {
			get {
				return signal;
			}
		}

		public Vector Output {
			get {
				return output;
			}
		}
		#endregion
	}
}

