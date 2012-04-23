using System;
using Gtk;

public partial class MainWindow: Gtk.Window
{		
	Image image;
	
	public MainWindow (): base (Gtk.WindowType.Toplevel)
	{
		Build ();
	}
	
	protected void OnDeleteEvent (object sender, DeleteEventArgs a)
	{
		Application.Quit ();
		a.RetVal = true;
	}

	protected void Quit (object sender, System.EventArgs e)
	{
		Application.Quit();
	}
	
	
	protected void Open (object sender, System.EventArgs e)
	{
		
	}
	
	protected void OpenFile (object sender, System.EventArgs e)
	{
		var paths = imageFileChooserButton.Filenames;
		LoadImage(paths[0]);
	}
	
	void LoadImage(string filename)
	{
		image = new Image(filename);
		sourceImage.Pixbuf = image.Pixbuf;
	}
}
