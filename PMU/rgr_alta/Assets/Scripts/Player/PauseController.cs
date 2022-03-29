using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class PauseController : MonoBehaviour
{
    [SerializeField]
    private GameObject pausePrefab;

    private GameObject pause;

    public void onPause()
    {
        pause = Instantiate(pausePrefab, transform);
        pause.transform.Find("You Died").gameObject.SetActive(false);
        
        Transform restart = pause.transform.Find("Restart");
        restart.GetComponent<Button>().onClick.RemoveAllListeners();
        restart.GetComponent<Button>().onClick.AddListener(onResume);
        restart.transform.Find("Text").GetComponent<Text>().text = "Продолжить";
        
        Time.timeScale = 0f;
    }

    public void onResume()
    {
        Time.timeScale = 1f;
        Destroy(pause);
    }

    public void onRestart()
    {
        Time.timeScale = 1f;
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    public void onMenu()
    {
        Time.timeScale = 1f;
        SceneManager.LoadScene(0);
    }
}
