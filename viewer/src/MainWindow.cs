using System;
using System.IO;
using Gtk;
using Gdk;

using Nanon.Math.Linear;
using Nanon.FileFormat;

public partial class MainWindow: Gtk.Window
{

	public MainWindow (): base (Gtk.WindowType.Toplevel)
	{
		Build ();
		DrawMatrix(CurrentMatrix);	
	}
	
	protected void OnDeleteEvent (object sender, DeleteEventArgs a)
	{
		Application.Quit ();
		a.RetVal = true;
	}
	
	protected void act (object sender, System.EventArgs e)
	{
		Application.Quit();
	}
	
	Matrix[] matrices;
	bool rowWiseView = true;
	
	protected void fileSelected (object sender, System.EventArgs e)
	{
		var paths = filechooserwidget.Filenames;
		string path = "";
	    if (paths.Length > 0)
		  path = paths[0];
		else return;
	
		matrices = null;
		matrices = Matrix.FromFile(path);
		SetupSpinButton();
		DrawMatrix(CurrentMatrix);	
	}
	
	void SetupSpinButton() {
		spinbutton.Sensitive = matrices.Length != 0;
		spinbutton.Adjustment = new Adjustment(0, 0, matrices.Length, 1, 10, 10);
	}
	
	Matrix CurrentMatrix 
	{
		get	
		{
			if (spinbutton.Sensitive)
				 return matrices[(int)spinbutton.Value];
			else return null;
		}
	}
	
	public byte[] ToBitmap(Matrix matrix, bool rowWise) {
		var bitmap = new byte[matrix.Size * 3];
		
		if (rowWise)
		{
			var i = 0;
			foreach(var x in matrix.Cells) {
				var v = (byte)x;
				bitmap[i]   = v;
				bitmap[i+1] = v;
				bitmap[i+2] = v;
				i += 3;
			}
		} else {
			var height = matrix.Height;
			var width  = matrix.Width;
			
			for (var j = 0; j < height; ++j)
				for (var i = 0; i < width; ++i)
				{	
					var si = i + j * width;
					var ti = 3 * (i * height + j);
					var v = (byte)matrix[si];
					bitmap[ti]     = v;
					bitmap[ti + 1] = v;
					bitmap[ti + 2] = v;
				}		
		}
		
		return bitmap;
	}
	
	void DrawMatrix(Matrix matrix) 
	{		
		if (matrix == null)
			return;
		
		var buffer = ToBitmap(matrix, rowWiseView);
		var pb = new Pixbuf(buffer, false, 8, 
		                       matrix.Width, 
		                       matrix.Height, matrix.Width * 3);
		matrixImage.Pixbuf = pb.ScaleSimple(300, 300, InterpType.Nearest);
	}

	protected void SpinChanged (object sender, System.EventArgs e)
	{
		DrawMatrix(CurrentMatrix);		
	}

	protected void ModeToggled (object sender, System.EventArgs e)
	{
		rowWiseView = !rowWiseView;
		DrawMatrix(CurrentMatrix);		
	}
}
