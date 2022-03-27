using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuController : MonoBehaviour
{
    public void onStart()
    {
        SceneManager.LoadScene(1);
    }
   
    public void onSettings()
    {

    }

    public void onExit()
    {
        Application.Quit();
    }
}
