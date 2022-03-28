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
    private Transform despawnPoint;

    void Start()
    {
        currentModules.Add(SpawnModule(modules.First(), transform));
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
                currentModules.Add(SpawnModule(modules[Random.Range(0, modules.Count)], transform, start));
            }
        }
    }

    private GameObject SpawnModule(GameObject prefab, Transform parent, Transform spawnPosition = null)
    {
        GameObject newModule = Instantiate(prefab, parent);
        
        if (spawnPosition != null)
        {
            newModule.transform.position = spawnPosition.transform.position;
        }
        
        return newModule;
    }
}
