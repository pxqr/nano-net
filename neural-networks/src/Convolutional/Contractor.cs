using System;

using Nanon.Math.Activator;
using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer
{
	public class Contractor : ISingleLayer<Matrix, Vector>
	{
		IActivator activator;
		
		public Contractor ()
		{
			
		}

		#region ISingleLayer[Matrix,Vector] implementation
		
		public Vector FeedForward (Matrix input)
		{
			throw new NotImplementedException ();
		}

		public Matrix PropagateBackward (Matrix input, Matrix predSignal, Vector error)
		{
			throw new NotImplementedException ();
		}

		public void FindGradient (Matrix input, Vector outputError)
		{
			throw new NotImplementedException ();
		}

		public Nanon.Math.Linear.Vector Gradient ()
		{
			throw new NotImplementedException ();
		}

		public void Correct (Nanon.Math.Linear.Vector gradients)
		{
			throw new NotImplementedException ();
		}

		public Nanon.Math.Linear.Vector ZeroGradients ()
		{
			throw new NotImplementedException ();
		}

		public Vector Signal {
			get {
				throw new NotImplementedException ();
			}
		}

		public Vector Output {
			get {
				throw new NotImplementedException ();
			}
		}
		#endregion
	}
}

