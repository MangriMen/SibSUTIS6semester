using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LevelController : MonoBehaviour
{
    public float speed = 3f;

    List<GameObject> currentModules = new List<GameObject>();

    [SerializeField]
    private List<GameObject> modules = new List<GameObject>();
    [SerializeField]
    private List<GameObject> iceModules = new List<GameObject>();

    [SerializeField]
    private Transform despawnPoint;

    void Start()
    {
        currentModules.Add(SpawnModule(modules.First(), transform, transform.position));
    }

    void Update()
    {
        GameObject toRemove = null;
        foreach(GameObject module in currentModules)
        {
            module.transform.Translate(Vector3.down * speed * Time.deltaTime);
            if (module.transform.Find("Start").transform.position.y <= despawnPoint.transform.position.y)
            {
                Destroy(module);
                toRemove = module;
            }
        }
        if (toRemove != null)
        {
            currentModules.Remove(toRemove);
        }

        if (currentModules.Count <= 1)
        {
            GameObject bottomModule = currentModules.Last();
            Transform start = bottomModule.transform.Find("Start");
            Transform end = bottomModule.transform.Find("End");

            if (end.transform.position.y <= despawnPoint.transform.position.y)
            {
                currentModules.Add(SpawnModule(modules[Random.Range(0, modules.Count)], transform, start.transform.position));
                if (Random.Range(0, 100) > 40)
                {
                    SpawnModule(iceModules[Random.Range(0, iceModules.Count)], transform, new Vector3(Random.Range(-1f, 1.1f), 7.32f));
                }
            }
        }
    }

    private GameObject SpawnModule(GameObject prefab, Transform parent, Vector3 spawnPosition)
    {
        GameObject newModule = Instantiate(prefab, parent);
        
        if (spawnPosition != null)
        {
            newModule.transform.position = spawnPosition;
        }
        
        return newModule;
    }
}
