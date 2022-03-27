using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovingLevel : MonoBehaviour
{
    public float speed = 3f;
    public float speedIncrease = 0f;
    public bool isRestart = false;
    public Transform endMark;

    [SerializeField]
    private SpriteRenderer sprite;

    private float positionY;
    private Vector2 restartPos;

    private void Awake()
    {
        restartPos = transform.position;
        positionY = sprite.bounds.size.y / 2 - restartPos.y;
    }

    void Update()
    {
        if (!isRestart)
        {
            if (!sprite.isVisible || (endMark && transform.position.y - sprite.bounds.size.y / 2 <= endMark.position.y))
            {
                return;
            }
        }
        transform.Translate(Vector3.down * speed * Time.deltaTime);
        if (isRestart && transform.position.y <= -positionY)
        {
            transform.position = restartPos;
        }
        speed += speedIncrease * Time.deltaTime;
    }

}
